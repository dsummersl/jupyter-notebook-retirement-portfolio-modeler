import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from notebooks.plugins.constants import LifePhase, LifePhases
from notebooks.plugins.modeler import run_multi_asset_simulation
from notebooks.plugins.base_asset import BaseAsset
import polars as pl


def test_run_multi_asset_simulation_no_assets():
    """ Test a simulation with no assets - should do nothing (there are no assets to increase or simulate over time). """
    life_phases_config = [
        {
            "name": "Working Years",
            "age": 30,
            "annual_income": 100000,
            "annual_expenses": 80000,
            "annual_investment": 20000,
        }
    ]

    num_years = 1
    num_simulations = 2

    total_df, individual_dfs, final_values, investment_fn, draw_fn = run_multi_asset_simulation(
        LifePhases(life_phases=life_phases_config).life_phases, num_years, num_simulations, 0.03
    )

    # Check output types
    assert isinstance(total_df, pl.DataFrame)
    assert isinstance(individual_dfs, dict)
    assert isinstance(final_values, np.ndarray)

    # Check DataFrame structure
    expected_columns = [
        "trading_day",
        "simulation",
        "portfolio_value",
        "trading_date",
        "investment",
        "withdrawal",
    ]
    assert total_df.columns == expected_columns

    # Check dimensions
    num_days = num_years * 252
    assert len(total_df) == num_simulations * num_days
    assert len(individual_dfs) == 0
    assert len(final_values) == num_simulations

    assert total_df["portfolio_value"].sum() == 0
    assert total_df["investment"].sum() == 0.0
    assert total_df["withdrawal"].sum() == 0


@pytest.mark.parametrize("asset_config", [
    {
        "type": "basic_asset",
        "params": {
            "initial_investment": 1000.0,
            "expected_return": 0.05,
            "volatility": 0.01,
        },
    },
    {
        "type": "stock_portfolio",
        "params": {
            "initial_investment": 1000,
            "portfolio_mix": {
              "VFIAX": 1.0
            }
        },
    },
    {
        "type": "mortgaged_real_estate",
        "params": {
            "property_value": 300000,
            "down_payment": 60000,
            "mortgage_rate": 0.04,
            "mortgage_term_years": 30,
            "expected_return": 0.03,
            "volatility": 0.01,
        },
    },
])
def test_run_multi_asset_simulation_with_asset_type(asset_config):
    """Test simulation with a parameterized asset type added via action."""
    life_phases_config = [
        {
            "name": "Working Years",
            "age": 30,
            "annual_income": 100000,
            "annual_expenses": 80000,
            "annual_investment": 20000,
            "actions": [
                {
                    "type": "grant_asset",
                    "name": "savings",
                    "config": asset_config
                }
            ],
        }
    ]

    num_years = 1
    num_simulations = 2

    total_df, individual_dfs, final_values, investment_fn, draw_fn = run_multi_asset_simulation(
        LifePhases(life_phases=life_phases_config).life_phases, num_years, num_simulations, 0.03
    )

    # Check output types
    assert isinstance(total_df, pl.DataFrame)
    assert isinstance(individual_dfs, dict)
    assert isinstance(final_values, np.ndarray)

    # Should have one individual asset
    assert len(individual_dfs) == 1
    assert "savings" in individual_dfs

    # Check that the asset values are non-zero
    assert individual_dfs["savings"]["savings_value"].sum() > 0
    assert total_df["portfolio_value"].sum() > 0


def test_buy_asset_causes_simulation_failure():
    """Test that when buy_asset cost exceeds available funds, simulation reaches zero and fails."""
    life_phases_config = [
        {
            "name": "Working Years",
            "age": 30,
            "annual_income": 0,
            "annual_expenses": 0,
            "annual_investment": 0,
            "actions": [
                {
                    "type": "grant_asset",
                    "name": "savings",
                    "cost": 0,
                    "config": {
                        "type": "basic_asset",
                        "params": {
                            "initial_investment": 5000.0,  # Only $5k available
                            "expected_return": 0.05,
                            "volatility": 0.01,
                        },
                    },
                },
                {
                    "type": "buy_asset",
                    "name": "house_down_payment",
                    "cost": 10000.0,
                    "funding_priority": ["savings"],
                    "config": {
                        "type": "mortgaged_real_estate",
                        "params": {
                            "property_value": 200000,
                            "down_payment": 10000,
                            "mortgage_rate": 0.04,
                            "mortgage_term_years": 30,
                            "expected_return": 0.03,
                            "volatility": 0.01,
                        }
                    },
                },
            ],
        }
    ]

    num_years = 1
    num_simulations = 1

    total_df, individual_dfs, final_values, investment_fn, draw_fn = run_multi_asset_simulation(
        LifePhases(life_phases=life_phases_config).life_phases, num_years, num_simulations, 0.03
    )

    # Verify the simulation started with savings
    assert "savings" in individual_dfs

    # Since the savings asset was completely drained (withdrawn $5k from $5k available),
    # the savings should be at or near 0
    savings_first_value = individual_dfs["savings"]["savings_value"][0]
    assert savings_first_value == pytest.approx(0.0, abs=1.0), "Savings should be depleted after failed buy_asset"

    # The house_down_payment asset should NOT exist because there were insufficient funds
    assert "house_down_payment" not in individual_dfs, "House should not be created with insufficient funds"

    # The final portfolio value should be at or near $0 since all funds were lost
    # and the simulation should have stopped early due to portfolio depletion
    assert final_values[0] == pytest.approx(0.0, abs=1.0), "Portfolio should be depleted after failed purchase"


def test_buy_asset_debits_source_asset_correctly():
    """Test that when buy_asset succeeds, the cost is properly debited from the source asset."""
    life_phases_config = [
        {
            "name": "Working Years",
            "age": 30,
            "annual_income": 0,
            "annual_expenses": 0,
            "annual_investment": 0,
            "actions": [
                {
                    "type": "grant_asset",
                    "name": "savings",
                    "cost": 0,
                    "config": {
                        "type": "basic_asset",
                        "params": {
                            "initial_investment": 50000.0,  # $50k available
                            "expected_return": 0.05,
                            "volatility": 0.01,
                        },
                    },
                },
                {
                    "type": "buy_asset",
                    "name": "house_down_payment",
                    "cost": 20000.0,  # Buying $20k asset
                    "funding_priority": ["savings"],
                    "config": {
                        "type": "basic_asset",
                        "params": {
                            "initial_investment": 0,  # Will be overwritten
                            "expected_return": 0.05,
                            "volatility": 0.01,
                        },
                    },
                },
            ],
        }
    ]

    num_years = 1
    num_simulations = 1

    total_df, individual_dfs, final_values, investment_fn, draw_fn = run_multi_asset_simulation(
        LifePhases(life_phases=life_phases_config).life_phases, num_years, num_simulations, 0.03
    )

    # Verify both assets exist
    assert "savings" in individual_dfs
    assert "house_down_payment" in individual_dfs

    # Get the first value (right after the buy_asset action on day 0)
    savings_first_value = individual_dfs["savings"]["savings_value"][0]
    house_first_value = individual_dfs["house_down_payment"]["house_down_payment_value"][0]

    # Savings should have been reduced by the cost
    expected_savings = 50000.0 - 20000.0
    assert savings_first_value == pytest.approx(expected_savings, abs=100.0), \
        f"Savings should be ${expected_savings} after buying asset (was {savings_first_value})"

    # House down payment should have the withdrawn amount
    assert house_first_value == pytest.approx(20000.0, abs=100.0), \
        f"House should have $20k initial investment (was {house_first_value})"

    # Total portfolio value should equal original investment (minus small transaction variations)
    total_first_value = total_df["portfolio_value"][0]
    assert total_first_value == pytest.approx(50000.0, abs=100.0), \
        f"Total portfolio should remain ~$50k (was {total_first_value})"

    # Final values should be roughly the initial investment plus growth
    assert final_values[0] > 45000, "Portfolio should have maintained value throughout simulation"
