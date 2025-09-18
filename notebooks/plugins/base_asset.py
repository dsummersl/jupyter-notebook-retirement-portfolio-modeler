from abc import ABC
import numpy as np
from .constants import num_trading_days, BasicAssetParams


class BaseAsset(ABC):
    """A plain asset, with explicitly defined initial_investment, expected_return, and volatility."""

    def __init__(self, params: dict):
        self.asset_params = BasicAssetParams(**params)
        self.value = self.asset_params.initial_investment

    def apply_monte_carlo(self, trade_day: int) -> bool:
        """
        Apply one value update step using a GBM-like process scaled by the configured frequency.

        Returns true if GBM was applied.
        """
        if self.value <= 0:
            return

        freq = str(self.asset_params.process_frequency).lower()

        # Determine if we should apply the GBM step on this trade_day
        apply_gbm = False
        if freq == "daily":
            dt = 1.0 / float(num_trading_days)
            apply_gbm = True
        elif freq == "monthly":
            dt = 1.0 / 12.0
            # Apply only on the first trading day of each month
            # Assumes trade_day starts at 1
            apply_gbm = (trade_day - 1) % (num_trading_days // 12) == 0
        else:
            dt = 1.0
            # Apply only on the first trading day of each year
            apply_gbm = (trade_day - 1) % num_trading_days == 0

        if not apply_gbm:
            return False

        # Geometric Brownian Motion step on value
        random_shock = np.random.normal(0.0, 1.0)
        step_return = (
            self.asset_params.expected_return * dt
            + self.asset_params.volatility * np.sqrt(dt) * random_shock
        )

        self.value *= float(np.exp(step_return))

        return True

    def get_current_value(self) -> float:
        """Return the current market value of the asset."""
        return self.value

    def deposit(self, amount: float):
        """Adds funds to the asset's value."""
        if amount < 0:
            raise ValueError("Deposit amount cannot be negative.")
        self.value += amount

    def withdraw(self, amount: float) -> float:
        """
        Withdraw a certain amount from the asset.
        Returns the amount actually withdrawn.
        """
        withdrawn_amount = min(self.value, amount)
        self.value -= withdrawn_amount
        return withdrawn_amount

    def modify(self, updates: dict):
        """
        Modify asset parameters mid-simulation. Subclasses should override this.
        """
        print(
            "WARNING: Asset of type "
            f"'{type(self).__name__}' does not support modification. Updates ignored: {updates}"
        )
