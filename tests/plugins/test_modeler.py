import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from notebooks.plugins.constants import LifePhase, LifePhases
from notebooks.plugins.modeler import run_multi_asset_simulation
from notebooks.plugins.base_asset import BaseAsset
import polars as pl


def test_run_multi_asset_simulation_no_assets():
    """Test simulation with no assets and basic lifecycle functions."""
    life_phases_config = [
        {
            "name": "Working Years",
            "age": 30,
            "annual_income": 100000,
            "annual_expenses": 80000,
            "annual_investment": 20000,
            "investment_allocation": {"stocks": 1.0},
        }
    ]

    num_years = 1
    num_simulations = 2

    total_df, individual_dfs, final_values = run_multi_asset_simulation(
        LifePhases(life_phases=life_phases_config).life_phases,
        num_years,
        num_simulations,
        0.03
    )

    # Check output types
    assert isinstance(total_df, pl.DataFrame)
    assert isinstance(individual_dfs, dict)
    assert isinstance(final_values, np.ndarray)

    # Check DataFrame structure
    expected_columns = ["trading_day", "simulation", "portfolio_value", "trading_date", "investment", "withdrawal"]
    assert total_df.columns == expected_columns

    # Check dimensions
    num_days = num_years * 252
    assert len(total_df) == num_simulations * num_days
    assert len(individual_dfs) == 0
    assert len(final_values) == num_simulations

    # Portfolio value should be zero throughout (no assets)
    assert total_df["portfolio_value"].sum() == 0
    # Investments should be positive
    assert total_df["investment"].sum() > 0
    # Withdrawals should be zero
    assert total_df["withdrawal"].sum() == 0


@patch("notebooks.plugins.modeler.ASSET_CLASS_MAP", {
    "basic_asset": BaseAsset,
    "stock_portfolio": MagicMock(),
    "mortgaged_real_estate": MagicMock(),
})
def test_run_multi_asset_simulation_with_basic_asset():
    """Test simulation with a basic asset added via action."""
    life_phases_config = [
        {
            "name": "Working Years",
            "age": 30,
            "annual_income": 100000,
            "annual_expenses": 80000,
            "annual_investment": 20000,
            "investment_allocation": {"savings": 1.0},
            "actions": [
                {
                    "type": "buy_asset",
                    "name": "savings",
                    "cost": 0,
                    "funding_priority": [],
                    "config": {
                        "type": "basic_asset",
                        "params": {
                            "initial_investment": 1000.0,
                            "expected_return": 0.05,
                            "volatility": 0.01,
                            "process_frequency": "daily"
                        }
                    }
                }
            ]
        }
    ]

    num_years = 1
    num_simulations = 2

    total_df, individual_dfs, final_values = run_multi_asset_simulation(
        LifePhases(life_phases=life_phases_config).life_phases,
        num_years,
        num_simulations,
        0.03
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
