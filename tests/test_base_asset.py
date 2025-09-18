import pytest
import numpy as np
from notebooks.plugins.base_asset import BaseAsset
from notebooks.plugins.constants import num_trading_days

default_params = {
    "initial_investment": 1000.0,
    "expected_return": 0.07,
    "volatility": 0.15,
    "process_frequency": "daily"
}

# since we're pegging the random seed, we can also check the resulting value in a predictable way:
random_values_for_seed: dict[str, int] = {
    "daily": 1004,
    "monthly": 1027,
    "yearly": 1155
}

def test_base_asset_initialization():
    asset = BaseAsset(default_params)
    assert asset.get_current_value() == 1000.0


def test_base_asset_deposit():
    asset = BaseAsset(default_params)
    asset.deposit(500.0)
    assert asset.get_current_value() == 1500.0


def test_base_asset_deposit_negative_raises_error():
    asset = BaseAsset(default_params)
    with pytest.raises(ValueError):
        asset.deposit(-100.0)


def test_base_asset_withdraw():
    asset = BaseAsset(default_params)
    withdrawn = asset.withdraw(300.0)
    assert withdrawn == 300.0
    assert asset.get_current_value() == 700.0


def test_base_asset_withdraw_more_than_available():
    asset = BaseAsset(default_params)
    withdrawn = asset.withdraw(1500.0)
    assert withdrawn == 1000.0
    assert asset.get_current_value() == 0.0


def test_base_asset_apply_monte_carlo_daily():
    asset = BaseAsset(default_params)
    np.random.seed(42)
    result = asset.apply_monte_carlo(1)
    assert result is True
    assert int(asset.get_current_value()) == random_values_for_seed["daily"]


@pytest.mark.parametrize(
    "process_frequency, trade_day, expected_result",
    [
        ("daily", 1, True),
        ("daily", 2, True),  # daily should always apply
        ("monthly", 1, True),
        ("monthly", 2, False),
        ("monthly", num_trading_days / 12, False),
        ("monthly", num_trading_days / 12 + 1, True),
        ("yearly", 1, True),
        ("yearly", 2, False),
        ("yearly", num_trading_days, False),
        ("yearly", num_trading_days + 1, True),
    ]
)
def test_base_asset_apply_monte_carlo_parametrized(process_frequency, trade_day, expected_result):
    asset = BaseAsset({
        **default_params,
        "process_frequency": process_frequency,
    })
    np.random.seed(42)
    result = asset.apply_monte_carlo(trade_day)
    assert result is expected_result
    if expected_result:
        assert int(asset.get_current_value()) == random_values_for_seed[process_frequency]
    else:
        assert int(asset.get_current_value()) == 1000
