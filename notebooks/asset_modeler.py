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


import sys
from pathlib import Path

# Get the directory containing this notebook file
notebook_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd() / "notebooks"
sys.path.append(str(notebook_dir))

from plugins.constants import num_trading_days, LifePhases
from plugins.modeler import (
    run_multi_asset_simulation,
    generate_lifecycle_functions,
)

# %config InlineBackend.figure_format = "retina"

# %config InlineBackend.figure_format = "retina"
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

# Acceptable values: "linear" or "log"
y_axis_scale = "linear"

life_phases = [
    {
        "name": "Early Career",
        "age": 30,
        "annual_income": 90_000,
        "annual_expenses": 70_000,
        "annual_investment": 20_000,
        "investment_allocation": {"stocks": 1.0},
        "actions": [{
            'type': 'grant_asset',
            'name': 'savings',
            'config': {
                'type': 'basic_asset',
                'params': {
                    'initial_investment': 0,
                    'expected_return': 0.05,
                    'volatility': 0.15
                }
            }
        }]
    },
    {
        "name": "Retire",
        "age": 65,
        "annual_income": 20_000,
        "annual_investment": 0,
    },
]

# # Example: Load configuration from a YAML file
# import yaml
#
# with open("../sample_papermill_settings.yaml", "r") as f:
#     config = yaml.safe_load(f)
# description = config["description"]
# num_simulations = config["num_simulations"]
# life_phases = config["life_phases"]


# %% jupyter={"source_hidden": true}
validated_life_phases = LifePhases(life_phases=life_phases)

# %% jupyter={"source_hidden": true}
simulation_start_age: int = int(life_phases[0]["age"])
num_years = simulation_end_age - simulation_start_age
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


# Create the functions from our configuration
investment_fn, investment_allocation_fn, draw_fn = generate_lifecycle_functions(
    life_phases, inflation_rate
)

# Run the simulation with the generated functions
simulated_totals_df, simulated_assets_df, final_values = run_multi_asset_simulation(
    life_phases_config=life_phases,
    investment_fn=investment_fn,
    investment_allocation_fn=investment_allocation_fn,  # ADDED
    draw_fn=draw_fn,
    num_years=num_years,
    num_simulations=num_simulations,
)

# Determine the set of asset names present over the simulation from outputs only
asset_names = sorted(simulated_assets_df.keys())


# %% jupyter={"source_hidden": true}
def add_lifecycle_milestones(ax, show_legend=True):
    """
    Adds vertical lines for key lifecycle milestones to any plot.
    This function now dynamically reads from the life_phases config.

    Fix: compute milestone offsets relative to the first phase age,
    not cumulatively against the first phase every loop.
    """
    current_year = datetime.now().year
    milestone_lines = []

    if len(life_phases) <= 1:
        return

    palette = sns.color_palette("husl", len(life_phases) - 1)
    for i in range(1, len(life_phases)):  # start from the second phase
        offset_years = life_phases[i]["age"] - life_phases[0]["age"]
        line_date = datetime(current_year + offset_years, 1, 1)
        line = ax.axvline(
            line_date,
            color=palette[i - 1],
            linestyle="--",
            alpha=0.8,
            label=life_phases[i]["name"],
        )
        milestone_lines.append(line)

    if show_legend and milestone_lines:
        handles, labels = ax.get_legend_handles_labels()
        milestone_handles = milestone_lines
        milestone_labels = [line.get_label() for line in milestone_lines]

        if handles:
            ax.legend(handles + milestone_handles, labels + milestone_labels, loc="upper left")
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


def calculate_asset_statistics(assets_df, total_final_values):
    """Calculate percentiles, volatility, and approximate initial values using only simulation outputs."""
    stats = {}

    # Compute approximate initial total from assets' first simulated values
    initial_total = 0.0

    # Add individual asset stats and accumulate initial values
    for name, df in assets_df.items():
        # Detect the value column dynamically
        value_cols = [c for c in df.columns if c.endswith("_value")]
        if not value_cols:
            continue
        value_col = value_cols[0]

        # Final values per simulation
        final_values = (
            df.sort(["simulation", "trading_day"])
            .with_columns(pl.col(value_col).last().over("simulation").alias("_final"))
            .unique(subset=["simulation"])
            .select("_final")
            .to_series()
            .to_numpy()
        )

        # First values per simulation (may be zero before asset is introduced)
        first_values = (
            df.sort(["simulation", "trading_day"])
            .with_columns(pl.col(value_col).first().over("simulation").alias("_first"))
            .unique(subset=["simulation"])
            .select("_first")
            .to_series()
            .to_numpy()
        )

        initial_avg = float(np.mean(first_values)) if len(first_values) else 0.0
        initial_total += initial_avg

        stats[name.title()] = {
            "p25": np.percentile(final_values, 25),
            "median": np.median(final_values),
            "p75": np.percentile(final_values, 75),
            "p95": np.percentile(final_values, 95),
            "volatility": np.std(final_values),
            "initial": initial_avg,
            "prob_success": np.mean(final_values > 0),
            "perc_success": (final_values > 0).sum() / num_simulations,
        }

    # Add total portfolio stats
    stats["Total"] = {
        "p25": np.percentile(total_final_values, 25),
        "median": np.median(total_final_values),
        "p75": np.percentile(total_final_values, 75),
        "p95": np.percentile(total_final_values, 95),
        "volatility": np.std(total_final_values),
        "initial": initial_total,
        "perc_success": (total_final_values > 0).sum() / num_simulations,
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
asset_stats = calculate_asset_statistics(simulated_assets_df, final_values)

asset_stats_to_polars_df(asset_stats)

# %% [markdown]
# ## Visualizations

# %% [markdown]
# ### Total Portfolio Value Projection
#
# This chart shows the aggregated value of all asset classes over time. The shaded area represents the 75% prediction interval, showing the range of most likely outcomes.


# %% jupyter={"source_hidden": true}
def money_formatter(x, pos):
    if x < 1_000:
        return f"${x}"
    elif x < 1_000_000:
        return f"${int(x / 1_000)}k"
    else:
        return f"${int(x / 1_000_000)}M"


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
if y_axis_scale == "log":
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
else:
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0)

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
    # Find the value column that ends with '_value'
    value_cols = [c for c in df.columns if c.endswith("_value")]
    y_col = value_cols[0] if value_cols else df.columns[-2]
    sns.lineplot(
        ax=ax,
        data=df,
        x="trading_date",
        y=y_col,
        legend=False,
        linewidth=1,
        errorbar=("pi", pi),
    )
    if y_axis_scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
    else:
        ax.set_yscale("linear")
        ax.set_ylim(bottom=0)
    ax.set_title(f"Median asset value ({name})")
    ax.set_ylabel("Asset Value")
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
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
    """Create cumulative investment and draw series distributed evenly across trading days."""
    num_days = num_years * num_trading_days

    # Allocate per-day amounts by spreading each year's total across its trading days,
    # then build cumulative series so the graphs evolve smoothly by trading day.
    daily_investments = np.zeros(num_days)
    daily_draws = np.zeros(num_days)

    for day in range(num_days):
        year = day // num_trading_days  # Which simulation year this trading day belongs to

        # Annual amounts for this year
        annual_investment = investment_fn(year)
        annual_draw = draw_fn(year)

        # Distribute evenly across trading days to avoid yearly steps
        daily_investments[day] = annual_investment / num_trading_days
        daily_draws[day] = annual_draw / num_trading_days

    cumulative_investments = np.cumsum(daily_investments)
    cumulative_draws = np.cumsum(daily_draws)

    return cumulative_investments, cumulative_draws


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
if y_axis_scale == "log":
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=1)
else:
    ax1.set_yscale("linear")
    ax1.set_ylim(bottom=0)


ax1.yaxis.set_major_formatter(FuncFormatter(money_formatter))
ax1.set_ylabel("Dollars")
ax1.set_title("Median Portfolio Value")

plt.tight_layout()
plt.show()
