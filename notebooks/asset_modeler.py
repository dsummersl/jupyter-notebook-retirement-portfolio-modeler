# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Retirement Forecast

# %% jupyter={"source_hidden": true}
import polars as pl
import numpy as np
from great_tables import GT
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from IPython.display import Markdown, display

# %config InlineBackend.figure_format = "retina"
import sys
from pathlib import Path

# Get the directory containing this notebook file
notebook_dir = (
    Path(__file__).parent if "__file__" in globals() else Path.cwd() / "notebooks"
)
sys.path.append(str(notebook_dir))

from plugins.basic_asset import BasicAsset
from plugins.stock_asset import StockPortfolioAsset
from plugins.constants import num_trading_days
from plugins.actions import ACTION_HANDLER_MAP

# %config InlineBackend.figure_format = "retina"
ASSET_CLASS_MAP = {
    "basic_asset": BasicAsset,
    "stock_portfolio": StockPortfolioAsset,
}

sns.set_theme()

# %% [markdown]
# ## Simulation Configuration

# %% + tags=["parameters"]
# Description of the simulation
description = "A sample simulation of a 30 year old making modest contributions every year, working part time after 65, and retiring at 70."

# Source of the parameters (for documentation purposes)
parameter_source = "this notebook"

# Number of Monte Carlo simulations to run
num_simulations = 10
simulation_end_age = 90
inflation_rate = 0.032

# Define the person's financial lifecycle as a series of phases.
# The simulation will process these in order.
# All monetary values are in today's dollars and will be inflation-adjusted as simulations progress.
life_phases = [
    {
        "name": "Early Career - Aggressive",
        "age": 30,  # Age 30
        "annual_income": 90_000,
        "annual_expenses": 70_000,
        "annual_investment": 20_000,
        # Define where the investment goes
        "investment_allocation": {"stocks": 1.0},
    },
    {
        "name": "Buy Primary Residence",
        "age": 40,  # Age 40
        "annual_investment": 20_000,  # Still investing this year
        "investment_allocation": {"stocks": 1.0},
        "actions": [
            {
                "type": "buy_asset",
                "name": "primary_residence",
                "cost": 150_000,  # Down payment
                "funding_priority": ["stocks", "emergency_fund"],
                "config": {
                    "type": "basic_asset",
                    "params": {
                        "expected_return": 0.04,
                        "volatility": 0.05,
                    },
                },
            }
        ],
    },
    {
        "name": "Mid Career - De-risking",
        "age": 41,  # Age 41
        "annual_income": 180_000,
        "annual_expenses": 120_000,
        "annual_investment": 30_000,
        # Example of splitting investment
        "investment_allocation": {"stocks": 0.8, "emergency_fund": 0.2},
        "actions": [
            {
                "type": "modify_asset",
                "name": "stocks",
                "updates": {
                    "portfolio_mix": {
                        "VTI": 0.6,
                        "BND": 0.4,
                    }  # Shift to 60/40 stocks/bonds
                },
            }
        ],
    },
    {
        "name": "Sell Home & Retire",
        "age": 55,  # Age 55
        "actions": [
            {"type": "sell_asset", "name": "primary_residence", "destination": "stocks"}
        ],
    },
    {
        "name": "Retirement",
        "age": 56,  # Age 56
        "annual_income": 40_000,
        "annual_expenses": 130_000,
        "annual_investment": 0,
        # No investment, so no allocation needed
    },
]

# Which asset classes to draw from to cover expense shortfalls.
draw_priority = ["emergency_fund", "stocks", "real_estate"]

# Define each asset class with its initial value and risk/return parameters.
asset_classes = {
    "stocks": {
        "type": "stock_portfolio",
        "params": {
            "initial_investment": 100_000,
            "portfolio_mix": {
                "VFIAX": 0.06,
                "VIMAX": 0.08,
                "VMLUX": 0.18,
                "VTCLX": 0.39,
                "VTMGX": 0.13,
                "VTMSX": 0.09,
                "VWIUX": 0.12,
            },
        },
    },
    "real_estate": {
        "type": "basic_asset",
        "params": {
            "initial_investment": 250_000,
            "expected_return": 0.04,
            "volatility": 0.05,
        },
    },
    "emergency_fund": {
        "type": "basic_asset",
        "params": {
            "initial_investment": 10_000,
            "expected_return": 0.041,
            "volatility": 0.005,
        },
    },
}

initial_investment = sum(
    v["params"]["initial_investment"] for v in asset_classes.values()
)


# %% jupyter={"source_hidden": true}
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


initiazed_asset_classes = initialize_assets(asset_classes)

# %% jupyter={"source_hidden": true}
num_years = simulation_end_age - life_phases[0]["age"]
pi = 75

display(Markdown(description))
display(Markdown(f"Parameters sourced from: {parameter_source}"))

# %% jupyter={"source_hidden": true}
# These functions fetch historical data and calculate the accurate risk/return profile for the stock portfolio.


def generate_trading_days(start_year: int, num_years: int) -> list:
    """Generate actual trading days for the simulation period, excluding weekends and major holidays."""
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

    melted = df.unpivot(
        index=["trading_day"], variable_name="simulation", value_name=value_name
    )

    date_mapping = {i + 1: date for i, date in enumerate(realistic_trading_days)}
    melted = melted.with_columns(
        pl.col("trading_day").replace_strict(date_mapping).alias("trading_date"),
    )

    return melted


# %% jupyter={"source_hidden": true}
# These functions process the `life_phases` configuration to drive the simulation.


def compound_value(base_value, rate, years):
    """Calculates the future value of a single amount after n years of compounding."""
    return base_value * ((1 + rate) ** years)


def generate_lifecycle_functions(phases: list[dict], base_inflation_rate: float):
    """
    Processes the life_phases configuration and returns investment and draw functions
    for the simulation.
    """
    # Validate that ages are in increasing order
    ages = [phase.get("age", 0) for phase in phases]
    if ages != sorted(ages):
        raise ValueError("Life phases must be specified in chronological order by age")

    phase_start_years = []
    for phase in phases:
        phase_start_years.append(phase.get("age", 0))

    def get_phase_for_year(year: int):
        for i in range(len(phase_start_years) - 1, -1, -1):
            if year >= phase_start_years[i]:
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


# Create the functions from our configuration
investment_fn, investment_allocation_fn, draw_fn = generate_lifecycle_functions(
    life_phases, inflation_rate
)


def run_multi_asset_simulation(
    asset_classes_config,
    life_phases_config,
    investment_fn,
    investment_allocation_fn,
    draw_fn,
    num_years,
    num_trading_days,
    num_simulations,
    priority_order=None,
):
    """
    Multi-asset Monte Carlo simulation using a pluggable action system.
    """
    event_schedule = {}
    for phase in life_phases_config:
        if "actions" in phase:
            event_schedule[phase.get("age", 0)] = phase["actions"]

    num_days = num_years * num_trading_days
    days_per_month = num_trading_days // 12

    all_asset_names = set(asset_classes_config.keys())
    raw_sims = {name: np.zeros((num_simulations, num_days)) for name in all_asset_names}
    investments = np.zeros((num_simulations, num_days))
    withdrawals = np.zeros((num_simulations, num_days))

    for sim in range(num_simulations):
        assets = initialize_assets(asset_classes_config)

        for day in range(num_days):
            year = day // num_trading_days

            if day > 0 and day % num_trading_days == 0:
                if year in event_schedule:
                    for action in event_schedule[year]:
                        handler = ACTION_HANDLER_MAP.get(action["type"])
                        if handler:
                            added, removed = handler(assets, action, ASSET_CLASS_MAP)
                            for name in added:
                                if name not in all_asset_names:
                                    all_asset_names.add(name)
                                    raw_sims[name] = np.zeros(
                                        (num_simulations, num_days)
                                    )
                        else:
                            print(f"Warning: Unknown action type '{action['type']}'")

                # --- Annual Asset Value Processing ---
                for asset in assets.values():
                    asset.process_annual_step()

            # --- Monthly Cash Flow ---
            if day > 0 and day % days_per_month == 0:
                # Investments (FIXED: Use allocation logic)
                monthly_investment = investment_fn(year) / 12
                allocation = investment_allocation_fn(year)
                for asset_name, proportion in allocation.items():
                    if asset_name in assets:
                        investment_amount = monthly_investment * proportion
                        assets[asset_name].deposit(investment_amount)
                investments[sim, day] = monthly_investment

                # Withdrawals
                monthly_draw = draw_fn(year) / 12
                remaining_draw = monthly_draw
                for asset_name in (p for p in priority_order if p in assets):
                    if remaining_draw <= 0:
                        break
                    withdrawn = assets[asset_name].withdraw(remaining_draw)
                    remaining_draw -= withdrawn
                withdrawals[sim, day] = monthly_draw - remaining_draw

            # --- Daily Recording ---
            for name in all_asset_names:
                if name in assets:
                    raw_sims[name][sim, day] = assets[name].get_current_value()
                elif day > 0:
                    raw_sims[name][sim, day] = raw_sims[name][sim, day - 1]

    total_portfolio_raw = sum(raw_sims.values())
    total_melted_df = melt_raw_data(
        total_portfolio_raw, "portfolio_value", datetime.now().year, num_years
    )
    total_melted_df = total_melted_df.with_columns(
        [
            pl.Series("investment", investments.flatten()),
            pl.Series("withdrawal", withdrawals.flatten()),
        ]
    )

    # Create a fresh set of assets just for getting the final list of keys
    final_assets_list = initialize_assets(asset_classes_config).keys()
    individual_melted_dfs = {
        name: melt_raw_data(
            raw_sims[name], f"{name}_value", datetime.now().year, num_years
        )
        for name in final_assets_list
    }
    final_values = total_portfolio_raw[:, -1]

    return total_melted_df, individual_melted_dfs, final_values


# Run the simulation with the generated functions
simulated_totals_df, simulated_assets_df, final_values = run_multi_asset_simulation(
    asset_classes_config=asset_classes,  # CHANGED: Pass the config dict
    life_phases_config=life_phases,
    investment_fn=investment_fn,
    investment_allocation_fn=investment_allocation_fn,  # ADDED
    draw_fn=draw_fn,
    num_years=num_years,
    num_trading_days=num_trading_days,
    num_simulations=num_simulations,
    priority_order=draw_priority,
)

# This part must come BEFORE the simulation is run to show the true initial state.
# Since this is a notebook, let's create the description from the config.
initiazed_asset_classes_for_display = initialize_assets(
    asset_classes
)  # For display only

rows = []
for name, config in initiazed_asset_classes_for_display.items():
    rows.append(
        {
            "Name": name.title(),
            "Initial Investment": config.initial_investment,
            "Expected Return": config.expected_return,
            "Volatility": config.volatility,
        }
    )

# Create Polars DataFrame
asset_df = pl.DataFrame(rows)
summary_df = asset_df.select([
    pl.lit("Total").alias("Name"),
    pl.col("Initial Investment").sum().alias("Initial Investment"),
    pl.col("Expected Return").mean().alias("Expected Return"),
    pl.col("Volatility").mean().alias("Volatility"),
])

(
    GT(pl.concat([asset_df, summary_df], how="vertical"))
    .fmt_currency("Initial Investment", decimals=0)
    .fmt_percent("Expected Return", decimals=0)
    .fmt_percent("Volatility", decimals=0)
)


# %% jupyter={"source_hidden": true}
def add_lifecycle_milestones(ax, show_legend=True):
    """
    Adds vertical lines for key lifecycle milestones to any plot.
    This function now dynamically reads from the life_phases config.
    """
    current_year_dt = datetime.now()
    milestone_lines = []

    cumulative_years = 0
    for i, phase in enumerate(life_phases[:-1]):  # No line needed for the final phase
        cumulative_years += phase["age"] - life_phases[0]["age"]
        line_date = datetime(current_year_dt.year + cumulative_years, 1, 1)
        line = ax.axvline(
            line_date,
            color=sns.color_palette("husl", len(life_phases))[i],
            linestyle="--",
            alpha=0.8,
            label=life_phases[i + 1]['name'],
        )
        milestone_lines.append(line)

    if show_legend and milestone_lines:
        handles, labels = ax.get_legend_handles_labels()
        milestone_handles = milestone_lines
        milestone_labels = [line.get_label() for line in milestone_lines]

        if handles:
            ax.legend(
                handles + milestone_handles, labels + milestone_labels, loc="upper left"
            )
        else:
            ax.legend(milestone_handles, milestone_labels, loc="upper left")


# %% jupyter={"source_hidden": true}

display(
    Markdown(f"""
## Summary of Outcomes

The table below shows the results of running {num_simulations} simulations of the portfolio over {num_years} years.

The `Percent > 0` column indicates the percentage of simulations where the final value was greater than zero, which is a measure of success for the portfolio.

Portfolio managers typically aim for a success rate of 75% or higher.
""")
)

# %% jupyter={"source_hidden": true}


def calculate_asset_statistics(assets_df, total_final_values, asset_classes_config):
    """Calculate percentiles and volatility for each asset class and total"""
    stats = {}

    # Add total portfolio stats
    stats["Total"] = {
        "p25": np.percentile(total_final_values, 25),
        "median": np.median(total_final_values),
        "p75": np.percentile(total_final_values, 75),
        "p95": np.percentile(total_final_values, 95),
        "volatility": np.std(total_final_values),
        "initial": sum(v.initial_investment for v in asset_classes_config.values()),
        "perc_success": (total_final_values > 0).sum() / num_simulations,
    }

    # Add individual asset stats
    for name, df in assets_df.items():
        value_col = f"{name}_value"
        # Get the final value for each simulation (i.e., last trading_day per simulation)
        final_values = (
            df.sort(["simulation", "trading_day"])  # ensure sorted
            .with_columns(
                [pl.col(value_col).last().over("simulation").alias("final_value")]
            )
            .unique(subset=["simulation"])
            .select("final_value")
            .to_series()
            .to_numpy()
        )
        stats[name.title()] = {
            "p25": np.percentile(final_values, 25),
            "median": np.median(final_values),
            "p75": np.percentile(final_values, 75),
            "p95": np.percentile(final_values, 95),
            "volatility": np.std(final_values),
            "initial": asset_classes_config[name].initial_investment,
            "prob_success": np.mean(final_values > 0),
            "perc_success": (final_values > 0).sum() / num_simulations,
        }

    return stats


def asset_stats_to_polars_df(asset_stats):
    """Convert asset_stats dict to a Polars DataFrame with formatted columns using great_tables"""
    metrics = [
        ("Initial Investment", lambda s: s["initial"]),
        ("25th Percentile", lambda s: s["p25"]),
        ("Median (50th)", lambda s: s["median"]),
        ("75th Percentile", lambda s: s["p75"]),
        ("95th Percentile", lambda s: s["p95"]),
        ("Volatility (Std Dev)", lambda s: s["volatility"]),
        ("Percent > 0", lambda s: s["perc_success"]),
    ]

    assets = list(asset_stats.keys())

    # Build data dict as plain Python lists of scalars
    data = {"Metric": [m[0] for m in metrics]}
    for asset in assets:
        values = [fmt(asset_stats[asset]) for _, fmt in metrics]
        # Convert all numpy types to Python floats/ints
        values = [
            v.item()
            if isinstance(v, np.generic)
            else float(v)
            if isinstance(v, (int, float, np.floating))
            else v
            for v in values
        ]
        data[asset] = values

    # Create Polars DataFrame safely
    df = pl.DataFrame({k: list(v) for k, v in data.items()})

    # Format columns using great_tables
    return (
        GT(df)
        .fmt_currency(["Total"] + assets, decimals=0)
        .fmt_percent(["Total"] + assets, rows=[-1], decimals=0)
        .fmt_currency(["Total"] + assets, rows=[-2], decimals=0, pattern="±{x}")
    )


# Calculate statistics for all assets
asset_stats = calculate_asset_statistics(
    simulated_assets_df, final_values, initiazed_asset_classes
)

asset_stats_to_polars_df(asset_stats)

# %% [markdown]
# ## Visualizations

# %% [markdown]
# ### Total Portfolio Value Projection
#
# This chart shows the aggregated value of all asset classes over time. The shaded area represents the 75% prediction interval, showing the range of most likely outcomes.


# %% jupyter={"source_hidden": true}
def money_formatter(x, pos):
    return f"${x / 1_000_000:.1f}M"


fig, ax = plt.subplots(figsize=(14, 7))
sns.lineplot(
    ax=ax,
    data=simulated_totals_df,
    x="trading_date",
    y="portfolio_value",
    legend=False,
    linewidth=1,
    errorbar=("pi", pi),
)
add_lifecycle_milestones(ax)

# Mark significant life events
ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xlabel("Year")
plt.ylabel("Total Portfolio Value")
plt.title(f"Monte Carlo Simulation of Total Portfolio Value ({num_simulations} runs)")
plt.legend(loc="upper left")
plt.show()

# %% [markdown]
# ### Individual Asset Class Projections
#
# These charts show the median projection for each individual asset class, allowing for a deeper analysis of how each component contributes to the total portfolio.

# %% jupyter={"source_hidden": true}
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
axes = axes.flatten()

for i, (name, df) in enumerate(simulated_assets_df.items()):
    ax = axes[i]
    sns.lineplot(
        ax=ax,
        data=df,
        x="trading_date",
        y=df.columns[-2],  # The value column name is dynamic
        legend=False,
        linewidth=1,
        errorbar=("pi", pi),
    )
    ax.set_title(f"Median Projection for {name.title()}")
    ax.set_ylabel("Asset Value")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x / 1_000_000:.2f}M"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    add_lifecycle_milestones(ax, show_legend=False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Investment & Draws Over Time
#
# This chart shows the cumulative investments and draws over time,
# alongside the median portfolio value. It helps visualize how cash flows impact the overall portfolio.


# %% jupyter={"source_hidden": true}
def create_investment_draw_data(investment_fn, draw_fn, num_years, num_trading_days):
    """Creates yearly investment and draw data aligned with simulation timeline."""
    num_days = num_years * num_trading_days

    # Initialize arrays for yearly values
    yearly_investments = np.zeros(num_days)
    yearly_draws = np.zeros(num_days)

    # Calculate total investment and draw for each year, then apply to all days in that year
    for day in range(num_days):
        year = day // num_trading_days  # Which year this day belongs to

        # Get the annual amounts for this year
        annual_investment = investment_fn(year)
        annual_draw = draw_fn(year)

        # Apply the same yearly value to all days in this year
        yearly_investments[day] = annual_investment
        yearly_draws[day] = annual_draw

    return yearly_investments, yearly_draws


# Generate the investment and draw data
cumulative_investments, cumulative_draws = create_investment_draw_data(
    investment_fn, draw_fn, num_years, num_trading_days
)

# Create the draw_df using the same realistic trading dates as the main simulation
realistic_trading_days = generate_trading_days(datetime.now().year, num_years)
num_days = num_years * num_trading_days

# Ensure we have the right number of trading days
if len(realistic_trading_days) < num_days:
    last_date = realistic_trading_days[-1]
    additional_days = pd.bdate_range(
        start=last_date + timedelta(days=1),
        periods=num_days - len(realistic_trading_days),
        freq="B",
    )
    realistic_trading_days.extend([day.to_pydatetime() for day in additional_days])

realistic_trading_days = realistic_trading_days[:num_days]

# Create the draw_df in the format expected by your plotting code
draw_data = []
for i, date in enumerate(realistic_trading_days):
    draw_data.append(
        {
            "trading_date": date,
            "value": cumulative_investments[i],
            "variable": "Cumulative Investments",
        }
    )
    draw_data.append(
        {
            "trading_date": date,
            "value": cumulative_draws[i],
            "variable": "Cumulative Draws",
        }
    )

draw_df = pl.DataFrame(draw_data)

# Calculate median portfolio value for the top plot
median_portfolio_df = simulated_totals_df.group_by("trading_date").agg(
    pl.col("portfolio_value").median().alias("median_portfolio_value")
)

# %% jupyter={"source_hidden": true}
# Convert to Polars DataFrame
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 5))

sns.lineplot(ax=ax2, data=draw_df, x="trading_date", y="value", hue="variable")
add_lifecycle_milestones(ax2, show_legend=False)

# Format the x-axis to show breaks every year, and show money
plt.gca().yaxis.set_major_formatter(FuncFormatter(money_formatter))

# Add labels and title
ax2.set_xlabel("Years")
ax2.set_ylabel("Dollars")
ax2.set_title("Investment & Draws over time")

sns.lineplot(
    ax=ax1,
    data=median_portfolio_df,
    x="trading_date",
    y="median_portfolio_value",
)
add_lifecycle_milestones(ax1, show_legend=False)


def mil_formatter(x, pos):
    return f"${x / 1_000_000:,.1f} M"


ax1.yaxis.set_major_formatter(FuncFormatter(mil_formatter))
ax1.set_ylabel("Dollars")
ax1.set_title("Median Portfolio Value")

plt.tight_layout()
plt.show()
