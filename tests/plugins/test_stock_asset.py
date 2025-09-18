import pytest
import numpy as np
from unittest.mock import patch
from notebooks.plugins.stock_asset import StockPortfolioAsset, calculate_portfolio_params
import polars as pl

# Mock data for stock returns
mock_log_returns_data = {
    "date": ["2020-01-01", "2020-01-02"],
    "AAPL": [0.01, 0.02],
    "MSFT": [0.015, 0.025],
}


def mock_fetch_stock_data(tickers_tuple):
    df = pl.DataFrame(mock_log_returns_data)
    return df


def mock_calculate_portfolio_params(_log_returns_df, _portfolio_mix):
    return 0.07, 0.15


default_params = {
    "portfolio_mix": {"AAPL": 0.6, "MSFT": 0.4},
    "initial_investment": 1000.0,
    "process_frequency": "daily",
}


@patch("notebooks.plugins.stock_asset.fetch_stock_data", side_effect=mock_fetch_stock_data)
@patch(
    "notebooks.plugins.stock_asset.calculate_portfolio_params",
    side_effect=mock_calculate_portfolio_params,
)
def test_stock_portfolio_asset_initialization(mock_calc, mock_fetch):
    asset = StockPortfolioAsset(default_params)
    assert asset.get_current_value() == 1000.0
    assert asset.portfolio_mix == {"AAPL": 0.6, "MSFT": 0.4}


@patch("notebooks.plugins.stock_asset.fetch_stock_data", side_effect=mock_fetch_stock_data)
@patch(
    "notebooks.plugins.stock_asset.calculate_portfolio_params",
    side_effect=mock_calculate_portfolio_params,
)
def test_stock_portfolio_asset_deposit(mock_calc, mock_fetch):
    asset = StockPortfolioAsset(default_params)
    asset.deposit(500.0)
    assert asset.get_current_value() == 1500.0


@patch("notebooks.plugins.stock_asset.fetch_stock_data", side_effect=mock_fetch_stock_data)
@patch(
    "notebooks.plugins.stock_asset.calculate_portfolio_params",
    side_effect=mock_calculate_portfolio_params,
)
def test_stock_portfolio_asset_deposit_negative_raises_error(mock_calc, mock_fetch):
    asset = StockPortfolioAsset(default_params)
    with pytest.raises(ValueError):
        asset.deposit(-100.0)


@patch("notebooks.plugins.stock_asset.fetch_stock_data", side_effect=mock_fetch_stock_data)
@patch(
    "notebooks.plugins.stock_asset.calculate_portfolio_params",
    side_effect=mock_calculate_portfolio_params,
)
def test_stock_portfolio_asset_withdraw(mock_calc, mock_fetch):
    asset = StockPortfolioAsset(default_params)
    withdrawn = asset.withdraw(300.0)
    assert withdrawn == 300.0
    assert asset.get_current_value() == 700.0


@patch("notebooks.plugins.stock_asset.fetch_stock_data", side_effect=mock_fetch_stock_data)
@patch(
    "notebooks.plugins.stock_asset.calculate_portfolio_params",
    side_effect=mock_calculate_portfolio_params,
)
def test_stock_portfolio_asset_withdraw_more_than_available(mock_calc, mock_fetch):
    asset = StockPortfolioAsset(default_params)
    withdrawn = asset.withdraw(1500.0)
    assert withdrawn == 1000.0
    assert asset.get_current_value() == 0.0


@patch("notebooks.plugins.stock_asset.fetch_stock_data", side_effect=mock_fetch_stock_data)
@patch(
    "notebooks.plugins.stock_asset.calculate_portfolio_params",
    side_effect=mock_calculate_portfolio_params,
)
def test_stock_portfolio_asset_apply_monte_carlo_daily(mock_calc, mock_fetch):
    asset = StockPortfolioAsset(default_params)
    np.random.seed(42)
    result = asset.apply_monte_carlo(1)
    assert result is True
    # Value should change due to Monte Carlo simulation
    # Using the same expected value as base asset test since we mocked the return and volatility
    assert int(asset.get_current_value()) == 1004


@patch("notebooks.plugins.stock_asset.fetch_stock_data", side_effect=mock_fetch_stock_data)
@patch(
    "notebooks.plugins.stock_asset.calculate_portfolio_params",
    side_effect=mock_calculate_portfolio_params,
)
def test_stock_portfolio_asset_modify(mock_calc, mock_fetch):
    asset = StockPortfolioAsset(default_params)
    new_mix = {"AAPL": 0.5, "MSFT": 0.5}
    asset.modify({"portfolio_mix": new_mix})
    assert asset.portfolio_mix == new_mix


def test_calculate_portfolio_params_zero_returns():
    """Test portfolio parameters calculation with zero returns."""
    df = pl.DataFrame({
        "AAPL": [0.0, 0.0],
        "MSFT": [0.0, 0.0],
    })
    mix = {"AAPL": 0.6, "MSFT": 0.4}
    expected_return, volatility = calculate_portfolio_params(df, mix)
    assert expected_return == 0.0
    assert volatility == 0.0


def test_calculate_portfolio_params_single_stock():
    """Test portfolio parameters calculation with a single stock."""
    df = pl.DataFrame({
        "AAPL": [0.01, 0.02],
    })
    mix = {"AAPL": 1.0}
    expected_return, volatility = calculate_portfolio_params(df, mix)
    # Expected annualized return: mean([0.01,0.02]) * 252 = 0.015 * 252 = 3.78
    assert expected_return == pytest.approx(3.78, abs=0.01)
    # Expected volatility: sqrt(sample variance of daily returns * 252)
    # Sample variance of [0.01,0.02] is 0.00005 -> annualized variance: 0.00005*252=0.0126 -> volatility: sqrt(0.0126)≈0.11224
    assert volatility == pytest.approx(0.11224, abs=0.001)


def test_calculate_portfolio_params_two_stocks():
    """Test portfolio parameters calculation with two stocks."""
    df = pl.DataFrame({
        "AAPL": [0.01, 0.02],
        "MSFT": [0.03, 0.04],
    })
    mix = {"AAPL": 0.6, "MSFT": 0.4}
    expected_return, volatility = calculate_portfolio_params(df, mix)

    # Calculate expected annualized returns for each stock
    aapl_mean = (0.01 + 0.02) / 2 * 252
    msft_mean = (0.03 + 0.04) / 2 * 252
    portfolio_return = 0.6 * aapl_mean + 0.4 * msft_mean
    assert expected_return == pytest.approx(portfolio_return, abs=0.01)

    # For simplicity, we check that it's a positive number
    assert volatility > 0
