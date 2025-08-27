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
import yahooquery as yq
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from IPython.display import Markdown, display

# %config InlineBackend.figure_format = "retina"
sns.set_theme()

# %% + tags=["parameters"]
# Description of the simulation
description = "A sample simulation of a 30 year old making modest contributions every year, working part time after 65, and retiring at 70."

# Source of the parameters (for documentation purposes)
parameter_source = "this notebook"

# Number of Monte Carlo simulations to run
num_simulations = 1_000

# Age of the person at which the simulation starts
starting_age = 30

# Number of years to simulate (default: up to age 90)
num_years = 90 - starting_age

# Number of years remaining to invest additional money while working full time
fulltime_years = 65 - starting_age

# Annual amount invested in the portfolio while working full time
investment_per_year = 5_000

# Number of years from now with no portfolio draw (e.g., working part time)
parttime_years = 5

# Annual income expected during part-time work
parttime_income = 35_000

# Estimated annual social security income, assumed to start at age 70
social_security = 32_000

# Total expected annual draw in today's dollars from the portfolio after full-time work (net of taxes/investments)
expected_draw = 100_000

# Which asset classes to draw to cover expenses from draw_fn
draw_priority = ["emergency_fund", "stocks", "real_estate"]

# Define each asset class with its initial value and risk/return parameters.
# The stock parameters will be calculated from historical data in the next step.
asset_classes = {
    "stocks": {
        "initial_investment": 100_000,
        "portfolio_mix": {
            "VFIAX": 0.06, "VIMAX": 0.08, "VMLUX": 0.18,
            "VTCLX": 0.39, "VTMGX": 0.13, "VTMSX": 0.09, "VWIUX": 0.12,
        },
    },
    "real_estate": {
        "initial_investment": 250_000, # Value of real estate holdings
        "expected_return": 0.04,  # Assumed long-term appreciation
        "volatility": 0.05,       # Lower than stocks, illiquid
    },
    "emergency_fund": {
        "initial_investment": 10_000, # Cash for emergencies
        "expected_return": 0.041,  # Interest
        "volatility": 0.005,      # Near-zero volatility
    },
}

initial_investment = sum(v["initial_investment"] for v in asset_classes.values())

# %% jupyter={"source_hidden": true}
# Number of trading days in a year (used for daily returns)
num_trading_days = 252

years_to_seventy = 70 - starting_age

pi = 75

display(Markdown(description))
display(Markdown(f"Parameters sourced from: {parameter_source}"))

# %% jupyter={"source_hidden": true}
# These functions fetch historical data and calculate the accurate risk/return profile for the stock portfolio.

def fetch_stock_data(tickers: list[str]) -> pl.DataFrame:
    """Fetches historical data for a list of stock tickers and calculates daily log returns."""
    data = yq.Ticker(tickers)
    historical_data = data.history(start="2001-01-01", end="2020-12-31", interval="1d")

    if historical_data.empty:
        raise ValueError("No data found for the given stock tickers.")

    historical_data.reset_index(inplace=True)
    df = pl.from_pandas(historical_data)

    # Pivot to get tickers as columns and calculate log returns
    adjclose_df = df.pivot(index="date", on="symbol", values="adjclose")
    log_returns_df = adjclose_df.select(
        [pl.col("date")] + [
            np.log(pl.col(c) / pl.col(c).shift(1)).alias(c)
            for c in tickers
        ]
    ).drop_nulls()

    return log_returns_df

def calculate_portfolio_params(log_returns_df: pl.DataFrame, portfolio_mix: dict) -> tuple[float, float]:
    """Calculates the expected return and volatility for a portfolio mix."""
    tickers = list(portfolio_mix.keys())
    weights = np.array([portfolio_mix[ticker] for ticker in tickers])

    # Calculate annualized expected returns for each ticker
    expected_returns = log_returns_df.select(tickers).mean().to_numpy().flatten() * num_trading_days

    # Calculate annualized covariance matrix
    cov_matrix = log_returns_df.select(tickers).to_pandas().cov().to_numpy() * num_trading_days

    # Calculate total portfolio expected return
    portfolio_expected_return = np.sum(weights * expected_returns)

    # Calculate total portfolio variance and volatility
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)

    return portfolio_expected_return, portfolio_volatility


def generate_trading_days(start_year: int, num_years: int) -> list:
    """Generate actual trading days for the simulation period, excluding weekends and major holidays."""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(start_year + num_years, 12, 31)

    # Create business day range excluding US federal holidays
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=start_date, end=end_date)

    # Generate business days and remove holidays
    business_days = pd.bdate_range(start=start_date, end=end_date, freq='B')
    trading_days = business_days.drop(business_days.intersection(holidays))

    return [day.to_pydatetime() for day in trading_days]

def melt_raw_data(raw_data, value_name, start_year: int, num_years: int):
    """Enhanced version that maps trading days to realistic calendar dates."""
    num_days = num_years * num_trading_days
    trading_days_series = pl.Series("trading_day", np.arange(1, num_days + 1))

    # Generate realistic trading days
    realistic_trading_days = generate_trading_days(start_year, num_years)

    # Ensure we have enough trading days for our simulation
    if len(realistic_trading_days) < num_days:
        # If we need more days, extend with additional business days
        last_date = realistic_trading_days[-1]
        additional_days = pd.bdate_range(
            start=last_date + timedelta(days=1),
            periods=num_days - len(realistic_trading_days),
            freq='B'
        )
        realistic_trading_days.extend([day.to_pydatetime() for day in additional_days])

    # Truncate to exact number needed
    realistic_trading_days = realistic_trading_days[:num_days]

    df = pl.DataFrame(
        {f"sim_{i}": sim for i, sim in enumerate(raw_data)},
    ).with_columns(trading_days_series)

    melted = df.unpivot(index=["trading_day"], variable_name="simulation", value_name=value_name)

    # Map trading day indices to realistic dates
    date_mapping = {i + 1: date for i, date in enumerate(realistic_trading_days)}
    melted = melted.with_columns(
        pl.col("trading_day").replace_strict(date_mapping).alias("trading_date"),
    )

    return melted


# %% jupyter={"source_hidden": true}
# These functions define the financial lifecycle and run the simulations.
def compound_interest(year, capital, ir, bonus):
    if year == 0:
        return capital
    return compound_interest(year - 1, capital * (1 + ir) + bonus, ir, bonus)

def investment_fn(year):
    return investment_per_year if year < fulltime_years else 0

def draw_fn(year):
    """
    Calculate the expected draw from the portfolio based on the year. Assumptions being:
    - During full-time work, no draws are made.
    - During part-time work, the expected draw is reduced by the part-time income.
    - When not working, the expected draw is the full amount.
    - After age 70, social security income is added to the expected draw.
    """

    if year < fulltime_years:
        return 0

    inflation_rate = 0.032

    # During part-time years
    if year < parttime_years:
        return compound_interest(year, expected_draw - parttime_income, inflation_rate, 0)

    # After age 70 social security kicks in
    if year > years_to_seventy:
        return compound_interest(year, expected_draw - social_security, inflation_rate, 0)

    # Full retirement before social security
    return compound_interest(year, expected_draw, inflation_rate, 0)

def simulate_portfolio_gbm(
    initial_investment,
    expected_return,
    volatility,
    num_years,
    num_trading_days,
    rng=None
):
    """
    Generator for a single simulation path using GBM.
    Yields portfolio_value each day.
    """
    num_days = num_years * num_trading_days
    dt = 1 / num_trading_days
    value = initial_investment

    if rng is None:
        rng = np.random.default_rng()

    for day in range(num_days):
        # Random GBM step
        random_shock = rng.normal(0, 1)
        daily_return = expected_return * dt + volatility * np.sqrt(dt) * random_shock
        value *= np.exp(daily_return)

        yield day, value



def run_multi_asset_simulation(
    asset_classes_config,
    investment_fn,
    draw_fn,
    num_years,
    num_trading_days,
    num_simulations,
    priority_order=None,
):
    """
    Multi-asset Monte Carlo simulation with daily GBM growth,
    monthly withdrawals, and investment tracking.
    Returns (total_melted_df, individual_melted_dfs, final_values).
    """
    if priority_order is None:
        priority_order = list(asset_classes_config.keys())

    num_days = num_years * num_trading_days
    days_per_month = num_trading_days // 12

    # Storage for results
    raw_sims = {name: np.zeros((num_simulations, num_days)) for name in asset_classes_config}
    investments = np.zeros((num_simulations, num_days))
    withdrawals = np.zeros((num_simulations, num_days))


    # TODO parallelize this loop for performance?
    for sim in range(num_simulations):
        balances = {name: cfg["initial_investment"] for name, cfg in asset_classes_config.items()}

        # Initialize random generators per asset
        rngs = {name: np.random.default_rng() for name in asset_classes_config}

        for day in range(num_days):
            year = day // num_trading_days

            # Step balances with GBM returns applied to CURRENT balances
            for name, cfg in asset_classes_config.items():
                dt = 1 / num_trading_days
                random_shock = rngs[name].normal(0, 1)
                daily_return = cfg["expected_return"] * dt + cfg["volatility"] * np.sqrt(dt) * random_shock
                balances[name] *= np.exp(daily_return)

            daily_investment = 0.0
            daily_withdrawal = 0.0

            # Apply monthly flows
            if day % days_per_month == 0 and day > 0:
                # Investment
                daily_investment = investment_fn(year) / 12
                balances["stocks"] += daily_investment

                # Withdraw in priority order
                remaining = draw_fn(year) / 12
                daily_withdrawal = remaining
                for asset in priority_order:
                    take = min(balances[asset], remaining)
                    balances[asset] -= take
                    remaining -= take
                    if remaining <= 0:
                        break

            # Record results
            investments[sim, day] = daily_investment
            withdrawals[sim, day] = daily_withdrawal
            for name in asset_classes_config:
                raw_sims[name][sim, day] = balances[name]

    # Combine into total portfolio
    total_portfolio_raw = sum(raw_sims.values())

    # Melt using your existing function
    total_melted_df = melt_raw_data(total_portfolio_raw, "portfolio_value", datetime.now().year, num_years)

    # Add investment/withdrawal columns using Polars with_columns
    total_melted_df = total_melted_df.with_columns([
        pl.Series("investment", investments.flatten()),
        pl.Series("withdrawal", withdrawals.flatten())
    ])

    # Individual melted dfs
    individual_melted_dfs = {
        name: melt_raw_data(raw_sims[name], f"{name}_value", datetime.now().year, num_years)
        for name in asset_classes_config
    }
    final_values = total_portfolio_raw[:, -1]

    return total_melted_df, individual_melted_dfs, final_values

# %% jupyter={"source_hidden": true}
# Step 1: Calculate the actual risk/return for the stock portfolio
stock_tickers = list(asset_classes['stocks']['portfolio_mix'].keys())
stock_log_returns = fetch_stock_data(stock_tickers)
stock_return, stock_volatility = calculate_portfolio_params(stock_log_returns, asset_classes['stocks']['portfolio_mix'])

# Step 2: Update the asset class config with the calculated values
asset_classes['stocks']['expected_return'] = stock_return
asset_classes['stocks']['volatility'] = stock_volatility

# Step 3: Run the full multi-asset simulation
simulated_totals_df, simulated_assets_df, final_values = run_multi_asset_simulation(
    asset_classes_config=asset_classes,
    investment_fn=investment_fn,
    draw_fn=draw_fn,
    num_years=num_years,
    num_trading_days=num_trading_days,
    num_simulations=num_simulations,
    priority_order=draw_priority
)

initial_investment_description = "Assets at beginning of simulation:\n"
for name, config in asset_classes.items():
    initial_investment_description += f"- {name.title()}:\n"
    initial_investment_description += f"  - Initial Investment: ${config['initial_investment']:,.0f}\n"
    initial_investment_description += f"  - Expected Return: {config['expected_return']:.2%}\n"
    initial_investment_description += f"  - Volatility: {config['volatility']:.2%}\n"

display(Markdown(initial_investment_description))

# %% jupyter={"source_hidden": true}
def add_lifecycle_milestones(ax, show_legend=True):
    """
    Adds vertical lines for key lifecycle milestones to any plot.

    Args:
        ax: matplotlib axis object
        show_legend: bool, whether to show legend for the milestone lines
    """
    current_year = datetime.now().year

    # Add vertical lines for key milestones
    milestone_lines = []

    if fulltime_years > 0:
        line = ax.axvline(
            datetime(current_year + fulltime_years, 5, 5),
            color="blue",
            linestyle="--",
            alpha=0.7,
            label="Stop Full-Time Work"
        )
        milestone_lines.append(line)

    if parttime_years > 0:
        line = ax.axvline(
            datetime(current_year + parttime_years, 5, 5),
            color="green",
            linestyle="-.",
            alpha=0.7,
            label="Full Retirement"
        )
        milestone_lines.append(line)

    # Age 70 - Social Security starts
    if years_to_seventy < num_years:
        line = ax.axvline(
            datetime(current_year + years_to_seventy, 5, 5),
            color="red",
            linestyle=":",
            alpha=0.7,
            label="Age 70 (Social Security)"
        )
        milestone_lines.append(line)

    if show_legend and milestone_lines:
        # Get existing legend entries if any
        handles, labels = ax.get_legend_handles_labels()
        # Add milestone lines to legend
        milestone_handles = [line for line in milestone_lines]
        milestone_labels = [line.get_label() for line in milestone_lines]

        if handles:  # If there are existing legend entries
            ax.legend(handles + milestone_handles, labels + milestone_labels, loc="upper left")
        else:  # If no existing legend
            ax.legend(milestone_handles, milestone_labels, loc="upper left")

# %% jupyter={"source_hidden": true}

display(Markdown(f"""
## Summary of Outcomes

The table below shows the results of running {num_simulations} simulations of the portfolio over {num_years} years.

The `Percent > 0` column indicates the percentage of simulations where the final value was greater than zero, which is a measure of success for the portfolio.

Portfolio managers typically aim for a success rate of 75% or higher.
"""))

# %% jupyter={"source_hidden": true}

def calculate_asset_statistics(assets_df, total_final_values, asset_classes_config):
    """Calculate percentiles and volatility for each asset class and total"""
    stats = {}

    # Add total portfolio stats
    stats['Total'] = {
        'p25': np.percentile(total_final_values, 25),
        'median': np.median(total_final_values),
        'p75': np.percentile(total_final_values, 75),
        'p95': np.percentile(total_final_values, 95),
        'volatility': np.std(total_final_values),
        'initial': sum(v["initial_investment"] for v in asset_classes_config.values()),
        'perc_success': (total_final_values > 0).sum() / num_simulations
    }

    # Add individual asset stats
    for name, df in assets_df.items():
        value_col = f"{name}_value"
        # Get the final value for each simulation (i.e., last trading_day per simulation)
        final_values = (
            df.sort(["simulation", "trading_day"])  # ensure sorted
              .with_columns([
                  pl.col(value_col).last().over("simulation").alias("final_value")
              ])
              .unique(subset=["simulation"])
              .select("final_value")
              .to_series()
              .to_numpy()
        )
        stats[name.title()] = {
            'p25': np.percentile(final_values, 25),
            'median': np.median(final_values),
            'p75': np.percentile(final_values, 75),
            'p95': np.percentile(final_values, 95),
            'volatility': np.std(final_values),
            'initial': asset_classes_config[name]['initial_investment'],
            'prob_success': np.mean(final_values > 0),
            'perc_success': (final_values > 0).sum() / num_simulations
        }

    return stats

def asset_stats_to_polars_df(asset_stats):
    """Convert asset_stats dict to a Polars DataFrame with formatted columns using great_tables"""
    metrics = [
        ("Initial Investment", lambda s: s['initial']),
        ("25th Percentile", lambda s: s['p25']),
        ("Median (50th)", lambda s: s['median']),
        ("75th Percentile", lambda s: s['p75']),
        ("95th Percentile", lambda s: s['p95']),
        ("Volatility (Std Dev)", lambda s: s['volatility']),
        ("Percent > 0", lambda s: s['perc_success']),
    ]

    assets = list(asset_stats.keys())

    # Build data dict as plain Python lists of scalars
    data = {"Metric": [m[0] for m in metrics]}
    for asset in assets:
        values = [fmt(asset_stats[asset]) for _, fmt in metrics]
        # Convert all numpy types to Python floats/ints
        values = [v.item() if isinstance(v, np.generic) else float(v) if isinstance(v, (int,float,np.floating)) else v
                  for v in values]
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
asset_stats = calculate_asset_statistics(simulated_assets_df, final_values, asset_classes)

asset_stats_to_polars_df(asset_stats)

# %% [markdown]
# ## Visualizations

# %% [markdown]
# ### Total Portfolio Value Projection
#
# This chart shows the aggregated value of all asset classes over time. The shaded area represents the 75% prediction interval, showing the range of most likely outcomes.

# %% jupyter={"source_hidden": true}
def money_formatter(x, pos):
    return f"${x/1_000_000:.1f}M"

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
        y=df.columns[-2], # The value column name is dynamic
        legend=False,
        linewidth=1,
        errorbar=("pi", pi),
    )
    ax.set_title(f"Median Projection for {name.title()}")
    ax.set_ylabel("Asset Value")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x/1_000_000:.2f}M"))
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
        freq='B'
    )
    realistic_trading_days.extend([day.to_pydatetime() for day in additional_days])

realistic_trading_days = realistic_trading_days[:num_days]

# Create the draw_df in the format expected by your plotting code
draw_data = []
for i, date in enumerate(realistic_trading_days):
    draw_data.append({
        "trading_date": date,
        "value": cumulative_investments[i],
        "variable": "Cumulative Investments"
    })
    draw_data.append({
        "trading_date": date,
        "value": cumulative_draws[i],
        "variable": "Cumulative Draws"
    })

draw_df = pl.DataFrame(draw_data)

# Calculate median portfolio value for the top plot
median_portfolio_df = (
    simulated_totals_df
    .group_by("trading_date")
    .agg(pl.col("portfolio_value").median().alias("median_portfolio_value"))
)

# %% jupyter={"source_hidden": true}
# Convert to Polars DataFrame
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 5))

sns.lineplot(
    ax=ax2,
    data=draw_df,
    x="trading_date",
    y="value",
    hue="variable"
)
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
    return f"${x/1_000_000:,.1f} M"

ax1.yaxis.set_major_formatter(FuncFormatter(mil_formatter))
ax1.set_ylabel("Dollars")
ax1.set_title("Median Portfolio Value")

plt.tight_layout()
plt.show()
