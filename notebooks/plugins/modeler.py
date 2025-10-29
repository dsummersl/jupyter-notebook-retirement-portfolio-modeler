import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
import logging
from dataclasses import dataclass
from typing import Callable
from .base_asset import BaseAsset
from .stock_asset import StockPortfolioAsset
from .real_estate_asset import MortgagedRealEstateAsset
from .constants import AssetAction, LifePhase, num_trading_days
from .actions import ACTION_HANDLER_MAP

logger = logging.getLogger(__name__)

ASSET_CLASS_MAP = {
    "basic_asset": BaseAsset,
    "stock_portfolio": StockPortfolioAsset,
    "mortgaged_real_estate": MortgagedRealEstateAsset,
}


@dataclass
class SimulationContext:
    """Encapsulates all shared data and configuration for a multi-asset simulation."""
    # Configuration
    num_simulations: int
    num_days: int
    days_per_month: int

    # Schedules
    event_schedule: dict[int, list[AssetAction]]
    withdraw_order_schedule: dict[int, list[str]]

    # Lifecycle functions
    investment_fn: Callable[[int], float]
    investment_allocation_fn: Callable[[int], dict]
    draw_fn: Callable[[int], float]

    # Shared storage arrays (modified during simulation)
    all_asset_names: set[str]
    raw_sims: dict[str, np.ndarray]
    investments: np.ndarray
    withdrawals: np.ndarray


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


def generate_lifecycle_functions(phases: list[LifePhase], base_inflation_rate: float):
    """
    Processes the life_phases configuration and returns investment and draw functions
    for the simulation.

    Change: If a phase omits annual_income, annual_expenses, or annual_investment,
    the value carries forward from the most recent prior phase that specified it.
    """
    # Validate that ages are in increasing order
    ages = [phase.age for phase in phases]
    if ages != sorted(ages):
        raise ValueError("Life phases must be specified in chronological order by age")

    # Base age of the first phase; translate simulation year offsets to absolute age
    base_age = phases[0].age

    def get_phase_for_year(year: int):
        absolute_age = base_age + year
        for i in range(len(ages) - 1, -1, -1):
            if absolute_age >= ages[i]:
                return phases[i]
        return phases[0]

    def get_latest_value(year: int, key: str, default):
        """
        Find the most recent value for `key` at or before the current year (by absolute age).
        Returns `default` if no prior phase specified the key.
        """
        absolute_age = base_age + year
        latest = None
        for phase in phases:
            if phase.age <= absolute_age and getattr(phase, key) is not None:
                latest = getattr(phase, key)
        return latest if latest is not None else default

    def investment_fn(year: int) -> float:
        # Carry forward annual_investment; default to 0 if never defined
        return float(get_latest_value(year, "annual_investment", 0.0))

    def investment_allocation_fn(year: int) -> dict:
        # Carry forward allocation; if never defined, default to 100% stocks for backward compatibility
        return dict(get_latest_value(year, "investment_allocation", {"stocks": 1.0}))

    def draw_fn(year: int) -> float:
        # Carry forward income/expenses; default to 0 if never defined
        income = float(get_latest_value(year, "annual_income", 0.0))
        expenses = float(get_latest_value(year, "annual_expenses", 0.0))
        inflated_income = compound_value(income, base_inflation_rate, year)
        inflated_expenses = compound_value(expenses, base_inflation_rate, year)
        net_draw = max(0.0, inflated_expenses - inflated_income)
        return float(net_draw)

    return investment_fn, investment_allocation_fn, draw_fn


def process_yearly_events(
    ctx: SimulationContext,
    sim: int,
    year: int,
    assets: dict,
    withdraw_order: list[str],
) -> list[str]:
    """
    Process yearly events including withdrawal order updates and asset actions.

    Returns:
        Updated withdrawal order list
    """
    # Apply life-phase-defined withdraw_order at the start of the year, if any.
    if year in ctx.withdraw_order_schedule:
        withdraw_order = ctx.withdraw_order_schedule[year]
        logger.debug("phase-set withdrawal order to " + ", ".join(withdraw_order) if withdraw_order else "(empty)")

    if year in ctx.event_schedule:
        for action in ctx.event_schedule[year]:
            # TODO validate the action, this should already have happened...
            handler = ACTION_HANDLER_MAP.get(action.type)
            if handler:
                logger.info(
                    f"[sim={sim}] Y{year}: executing action {action.type} name={action.name}"
                )
                added, removed = handler(assets, action)
                logger.info(f"[sim={sim}] Y{year}: added={added} removed={removed}")
                for name in added:
                    if name not in ctx.all_asset_names:
                        ctx.all_asset_names.add(name)
                        ctx.raw_sims[name] = np.zeros((ctx.num_simulations, ctx.num_days))
            else:
                logger.warning(f"[sim={sim}] Unknown action type '{action.type}'")

    return withdraw_order


def process_monthly_operations(
    ctx: SimulationContext,
    sim: int,
    day: int,
    year: int,
    assets: dict,
    withdraw_order: list[str],
) -> None:
    """
    Process monthly investment deposits and withdrawals.

    Modifies assets, investments array, and withdrawals array in place.
    """
    # Handle monthly investments
    monthly_investment = ctx.investment_fn(year) / 12
    allocation = ctx.investment_allocation_fn(year)
    for asset_name, proportion in allocation.items():
        # If asset exists, deposit. Otherwise skip until created by action.
        if asset_name in assets:
            investment_amount = monthly_investment * proportion
            assets[asset_name].deposit(investment_amount)
            logger.info(
                f"[sim={sim}] D{day} Y{year}: deposit {investment_amount:.2f} -> {asset_name}"
            )
        else:
            logger.info(
                f"[sim={sim}] D{day} Y{year}: skip deposit {monthly_investment * proportion:.2f} -> missing asset '{asset_name}'"
            )
    ctx.investments[sim, day] = monthly_investment

    # Handle monthly withdrawals
    monthly_draw = ctx.draw_fn(year) / 12
    remaining_draw = monthly_draw

    # Determine withdrawal order based on maintained list of assets.
    # If no explicit order has been established yet, fall back to current assets order.
    order = withdraw_order if withdraw_order else list(assets.keys())
    # Append any missing assets to the end of the order
    missing_assets = [a for a in assets.keys() if a not in order]
    full_order = order + missing_assets
    logger.debug("order for withdrawal: " + ", ".join(full_order))
    logger.debug("withdraw_order for withdrawal: " + ", ".join(withdraw_order))
    for asset_name in (p for p in full_order if p in assets):
        if remaining_draw <= 0:
            break
        withdrawn = assets[asset_name].withdraw(remaining_draw)
        remaining_draw -= withdrawn
    fulfilled = monthly_draw - remaining_draw
    ctx.withdrawals[sim, day] = fulfilled
    logger.info(
        f"[sim={sim}] D{day} Y{year}: withdraw requested={monthly_draw:.2f} fulfilled={fulfilled:.2f}"
    )
    if remaining_draw > 0:
        logger.info(
            f"[sim={sim}] D{day} Y{year}: UNFULFILLED WITHDRAWAL of {remaining_draw:.2f} remaining!"
        )


def run_multi_asset_simulation(
    life_phases_config: list[LifePhase],
    num_years: int,
    num_simulations: int,
    inflation_rate: float
):
    """
    Multi-asset Monte Carlo simulation using a pluggable action system.

    Returns:
        tuple: (total_melted_df, individual_melted_dfs, final_values, investment_fn, draw_fn)
    """
    investment_fn, investment_allocation_fn, draw_fn = generate_lifecycle_functions(
        life_phases_config, inflation_rate
    )

    # Build an event schedule keyed by years since the start age
    event_schedule: dict[int, list[AssetAction]] = {}
    # Also capture any withdraw_order specified at the life phase level
    withdraw_order_schedule: dict[int, list[str]] = {}
    start_age = life_phases_config[0].age if life_phases_config else 0
    for phase in life_phases_config:
        rel_year = max(0, phase.age - start_age)
        if phase.actions:
            event_schedule[rel_year] = phase.actions
        # If the life phase specifies a withdraw_order, apply it at the start of that phase
        if phase.withdraw_order:
            withdraw_order_schedule[rel_year] = list(phase.withdraw_order)

    num_days = num_years * num_trading_days
    days_per_month = num_trading_days // 12

    # Create simulation context with all shared data
    ctx = SimulationContext(
        num_simulations=num_simulations,
        num_days=num_days,
        days_per_month=days_per_month,
        event_schedule=event_schedule,
        withdraw_order_schedule=withdraw_order_schedule,
        investment_fn=investment_fn,
        investment_allocation_fn=investment_allocation_fn,
        draw_fn=draw_fn,
        all_asset_names=set(),
        raw_sims={},
        investments=np.zeros((num_simulations, num_days)),
        withdrawals=np.zeros((num_simulations, num_days)),
    )

    # Dynamic withdrawal order; tracks assets as they are added/removed.
    # Actions may explicitly set it via 'withdraw_order'.
    withdraw_order: list[str] = []

    for sim in range(num_simulations):
        assets: dict = {}

        for day in range(num_days):
            year = day // num_trading_days
            is_start_of_year = day % num_trading_days == 0
            is_start_of_month = day > 0 and day % days_per_month == 0

            if is_start_of_year:
                withdraw_order = process_yearly_events(
                    ctx=ctx,
                    sim=sim,
                    year=year,
                    assets=assets,
                    withdraw_order=withdraw_order,
                )

            if is_start_of_month:
                process_monthly_operations(
                    ctx=ctx,
                    sim=sim,
                    day=day,
                    year=year,
                    assets=assets,
                    withdraw_order=withdraw_order,
                )

            for asset in assets.values():
                asset.apply_monte_carlo(day)

            for name in ctx.all_asset_names:
                if name in assets:
                    ctx.raw_sims[name][sim, day] = assets[name].get_current_value()
                elif day > 0:
                    ctx.raw_sims[name][sim, day] = ctx.raw_sims[name][sim, day - 1]

    if ctx.raw_sims:
        total_portfolio_raw = sum(ctx.raw_sims.values())
    else:
        total_portfolio_raw = np.zeros((num_simulations, num_days))
    total_melted_df = melt_raw_data(
        total_portfolio_raw, "portfolio_value", datetime.now().year, num_years
    )
    total_melted_df = total_melted_df.with_columns(
        [
            pl.Series("investment", ctx.investments.flatten()),
            pl.Series("withdrawal", ctx.withdrawals.flatten()),
        ]
    )

    # Build individual asset frames for all assets that existed
    individual_melted_dfs = {}
    for name in sorted(ctx.all_asset_names):
        individual_melted_dfs[name] = melt_raw_data(
            ctx.raw_sims[name], f"{name}_value", datetime.now().year, num_years
        )
    final_values = total_portfolio_raw[:, -1]

    return total_melted_df, individual_melted_dfs, final_values, ctx.investment_fn, ctx.draw_fn
