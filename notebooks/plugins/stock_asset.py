from .basic_asset import BasicAsset
from .constants import num_trading_days
import numpy as np
import yahooquery as yq
import polars as pl


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
        [pl.col("date")]
        + [np.log(pl.col(c) / pl.col(c).shift(1)).alias(c) for c in tickers]
    ).drop_nulls()

    return log_returns_df


def calculate_portfolio_params(
    log_returns_df: pl.DataFrame, portfolio_mix: dict
) -> tuple[float, float]:
    """Calculates the expected return and volatility for a portfolio mix."""
    tickers = list(portfolio_mix.keys())
    weights = np.array([portfolio_mix[ticker] for ticker in tickers])

    # Calculate annualized expected returns for each ticker
    expected_returns = (
        log_returns_df.select(tickers).mean().to_numpy().flatten() * num_trading_days
    )

    # Calculate annualized covariance matrix
    cov_matrix = (
        log_returns_df.select(tickers).to_pandas().cov().to_numpy() * num_trading_days
    )

    # Calculate total portfolio expected return
    portfolio_expected_return = np.sum(weights * expected_returns)

    # Calculate total portfolio variance and volatility
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)

    return portfolio_expected_return, portfolio_volatility


class StockPortfolioAsset(BasicAsset):
    """
    An asset representing a portfolio of stocks.
    It calculates its own return and volatility from the mix.
    """

    def __init__(self, params: dict):
        super().__init__(
            {**params, **self._calculate_portfolio_metrics(params["portfolio_mix"])}
        )

    def _calculate_portfolio_metrics(self, mix: dict):
        stock_tickers = list(mix.keys())
        stock_log_returns = fetch_stock_data(stock_tickers)
        stock_return, stock_volatility = calculate_portfolio_params(
            stock_log_returns, mix
        )

        return {
            "expected_return": stock_return,
            "volatility": stock_volatility,
        }
