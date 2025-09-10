import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
import logging
from logging.handlers import RotatingFileHandler

# Configure a rotating file logger for simulations
logger = logging.getLogger("simulation")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler("simulation.log", maxBytes=2_000_000, backupCount=3)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

from .basic_asset import BasicAsset
from .stock_asset import StockPortfolioAsset
from .constants import num_trading_days
from .actions import ACTION_HANDLER_MAP


ASSET_CLASS_MAP = {
    "basic_asset": BasicAsset,
    "stock_portfolio": StockPortfolioAsset,
}


def initialize_assets(config: dict) -> dict:
    """
    Initializes asset objects from the configuration dictionary
    using the dynamic asset class map.
    """
    assets = {}
    for name, asset_config in config.items():
        asset_type = asset_config["type"]
        asset_class = ASSET_CLASS_MAP.get(asset_type)

        if not asset_class:
            raise ValueError(f"Unknown asset type '{asset_type}' for asset '{name}'.")

        asset_config["params"]["name"] = name
        assets[name] = asset_class(asset_config["params"])

    return assets


def compound_value(base_value, rate, years):
    """Calculates the future value of a single amount after n years of compounding."""
    return base_value * ((1 + rate) ** years)


def generate_trading_days(start_year: int, num_years: int) -> list:
    """Generate actual trading days for the simulation period, excluding weekends and US federal holidays."""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(start_year + num_years, 12, 31)

    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=start_date, end=end_date)

    business_days = pd.bdate_range(start=start_date, end=end_date, freq="B")
    trading_days = business_days.drop(business_days.intersection(holidays))

    return [day.to_pydatetime() for day in trading_days]


def melt_raw_data(raw_data, value_name, start_year: int, num_years: int):
    """Enhanced version that maps trading days to realistic calendar dates."""
    num_days = num_years * num_trading_days
    trading_days_series = pl.Series("trading_day", np.arange(1, num_days + 1))

    realistic_trading_days = generate_trading_days(start_year, num_years)

    if len(realistic_trading_days) < num_days:
        last_date = realistic_trading_days[-1]
        additional_days = pd.bdate_range(
            start=last_date + timedelta(days=1),
            periods=num_days - len(realistic_trading_days),
            freq="B",
        )
        realistic_trading_days.extend([day.to_pydatetime() for day in additional_days])

    realistic_trading_days = realistic_trading_days[:num_days]

    df = pl.DataFrame(
        {f"sim_{i}": sim for i, sim in enumerate(raw_data)},
    ).with_columns(trading_days_series)

    melted = df.unpivot(index=["trading_day"], variable_name="simulation", value_name=value_name)

    date_mapping = {i + 1: date for i, date in enumerate(realistic_trading_days)}
    melted = melted.with_columns(
        pl.col("trading_day").replace_strict(date_mapping).alias("trading_date"),
    )

    return melted


def generate_lifecycle_functions(phases: list[dict], base_inflation_rate: float):
    """
    Processes the life_phases configuration and returns investment and draw functions
    for the simulation.
    """
    # Validate that ages are in increasing order
    ages = [phase.get("age", 0) for phase in phases]
    if ages != sorted(ages):
        raise ValueError("Life phases must be specified in chronological order by age")

    # Base age of the first phase; translate simulation year offsets to absolute age
    base_age = phases[0].get("age", 0)

    phase_start_years = []
    for phase in phases:
        phase_start_years.append(phase.get("age", 0))

    def get_phase_for_year(year: int):
        absolute_age = base_age + year
        for i in range(len(phase_start_years) - 1, -1, -1):
            if absolute_age >= phase_start_years[i]:
                return phases[i]
        return phases[0]

    def investment_fn(year: int) -> float:
        phase = get_phase_for_year(year)
        return phase.get("annual_investment", 0)

    def investment_allocation_fn(year: int) -> dict:
        phase = get_phase_for_year(year)
        # Default to 100% stocks if not specified, for backward compatibility
        return phase.get("investment_allocation", {"stocks": 1.0})

    def draw_fn(year: int) -> float:
        phase = get_phase_for_year(year)
        income = phase.get("annual_income", 0)
        expenses = phase.get("annual_expenses", 0)
        inflated_income = compound_value(income, base_inflation_rate, year)
        inflated_expenses = compound_value(expenses, base_inflation_rate, year)
        net_draw = max(0, inflated_expenses - inflated_income)
        return net_draw

    return investment_fn, investment_allocation_fn, draw_fn


def run_multi_asset_simulation(
    life_phases_config,
    investment_fn,
    investment_allocation_fn,
    draw_fn,
    num_years,
    num_simulations,
    priority_order=None,
):
    """
    Multi-asset Monte Carlo simulation using a pluggable action system.
    """
    # Build an event schedule keyed by years since the start age
    event_schedule: dict[int, list[dict]] = {}
    if not life_phases_config:
        life_phases_config = []
    start_age = life_phases_config[0].get("age", 0) if life_phases_config else 0
    for phase in life_phases_config:
        if "actions" in phase and phase["actions"]:
            rel_year = max(0, phase.get("age", start_age) - start_age)
            event_schedule[rel_year] = phase["actions"]

    num_days = num_years * num_trading_days
    days_per_month = num_trading_days // 12

    # Simulation storage
    all_asset_names: set[str] = set()
    raw_sims: dict[str, np.ndarray] = {}
    investments = np.zeros((num_simulations, num_days))
    withdrawals = np.zeros((num_simulations, num_days))

    for sim in range(num_simulations):
        # Start with no assets; actions and allocations will introduce them
        assets: dict = {}

        for day in range(num_days):
            year = day // num_trading_days

            if day % num_trading_days == 0:
                if year in event_schedule:
                    for action in event_schedule[year]:
                        handler = ACTION_HANDLER_MAP.get(action["type"])
                        if handler:
                            logger.info(f"[sim={sim}] Y{year}: executing action {action['type']} name={action.get('name')}")
                            added, removed = handler(assets, action, ASSET_CLASS_MAP)
                            logger.info(f"[sim={sim}] Y{year}: added={added} removed={removed}")
                            for name in added:
                                if name not in all_asset_names:
                                    all_asset_names.add(name)
                                    raw_sims[name] = np.zeros((num_simulations, num_days))
                        else:
                            logger.warning(f"[sim={sim}] Unknown action type '{action['type']}'")

                # --- Annual Asset Value Processing (after first full year) ---
                if day > 0:
                    for asset in assets.values():
                        asset.process_annual_step()
                    logger.info(f"[sim={sim}] Y{year}: processed annual step for {len(assets)} assets")

            # --- Monthly Cash Flow ---
            if day > 0 and day % days_per_month == 0:
                # Investments (FIXED: Use allocation logic)
                monthly_investment = investment_fn(year) / 12
                allocation = investment_allocation_fn(year)
                for asset_name, proportion in allocation.items():
                    # If asset exists, deposit. Otherwise skip until created by action.
                    if asset_name in assets:
                        investment_amount = monthly_investment * proportion
                        assets[asset_name].deposit(investment_amount)
                        logger.info(f"[sim={sim}] D{day} Y{year}: deposit {investment_amount:.2f} -> {asset_name}")
                    else:
                        logger.info(f"[sim={sim}] D{day} Y{year}: skip deposit {monthly_investment * proportion:.2f} -> missing asset '{asset_name}'")
                investments[sim, day] = monthly_investment

                # Withdrawals
                monthly_draw = draw_fn(year) / 12
                remaining_draw = monthly_draw
                withdrawal_order = (
                    priority_order
                    if isinstance(priority_order, (list, tuple))
                    else list(assets.keys())
                )
                for asset_name in (p for p in withdrawal_order if p in assets):
                    if remaining_draw <= 0:
                        break
                    withdrawn = assets[asset_name].withdraw(remaining_draw)
                    remaining_draw -= withdrawn
                fulfilled = monthly_draw - remaining_draw
                withdrawals[sim, day] = fulfilled
                logger.info(f"[sim={sim}] D{day} Y{year}: withdraw requested={monthly_draw:.2f} fulfilled={fulfilled:.2f}")

            # --- Daily Recording ---
            for name in all_asset_names:
                if name in assets:
                    raw_sims[name][sim, day] = assets[name].get_current_value()
                elif day > 0:
                    raw_sims[name][sim, day] = raw_sims[name][sim, day - 1]

    if raw_sims:
        total_portfolio_raw = sum(raw_sims.values())
    else:
        total_portfolio_raw = np.zeros((num_simulations, num_days))
    total_melted_df = melt_raw_data(
        total_portfolio_raw, "portfolio_value", datetime.now().year, num_years
    )
    total_melted_df = total_melted_df.with_columns(
        [
            pl.Series("investment", investments.flatten()),
            pl.Series("withdrawal", withdrawals.flatten()),
        ]
    )

    # Build individual asset frames for all assets that existed
    individual_melted_dfs = {}
    for name in sorted(all_asset_names):
        individual_melted_dfs[name] = melt_raw_data(
            raw_sims[name], f"{name}_value", datetime.now().year, num_years
        )
    final_values = total_portfolio_raw[:, -1]

    return total_melted_df, individual_melted_dfs, final_values
