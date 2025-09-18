import logging
from typing_extensions import override
from .base_asset import BaseAsset

logger = logging.getLogger(__name__)


def calculate_monthly_payment(principal, monthly_rate, nper):
    # Formula: P = (r*PV) / (1 - (1 + r)**-n)
    return (monthly_rate * principal) / (1 - (1 + monthly_rate) ** -nper)


class MortgagedRealEstateAsset(BaseAsset):
    """
    Represents a real estate property with a mortgage.
    Tracks property value appreciation and mortgage balance.
    'value' is the property value (not including mortgage).
    """

    def __init__(self, params: dict):
        super().__init__(
            {
                "initial_investment": params["property_value"],
                "expected_return": params["expected_return"],
                "volatility": params["volatility"],
                "process_frequency": "monthly",
            }
        )
        self.mortgage_balance = params["property_value"] - params["down_payment"]
        self.rate = params["mortgage_rate"]
        self.term = params["mortgage_term_years"]
        monthly_rate = self.rate / 12
        nper = self.term * 12
        self.monthly_payment = calculate_monthly_payment(self.mortgage_balance, monthly_rate, nper)

    @override
    def apply_monte_carlo(self, trade_day: int) -> bool:
        # Update the value of the property using GBM-like process monthly
        applied = super().apply_monte_carlo(trade_day)

        # when the monte carlo is applied monthly, subtract the portion of the monthly_payment that goes to the principal
        if applied and self.mortgage_balance > 0:
            interest = self.mortgage_balance * self.rate / 12
            principal_payment = self.monthly_payment - interest
            self.mortgage_balance -= principal_payment
            logger.debug(
                "D%1f: balance reduced by %.2f to %.2f, value %2.f",
                trade_day,
                principal_payment,
                self.mortgage_balance,
                self.value,
            )
        elif applied:
            logger.debug("Property value updated to %.2f", self.value)

        return applied

    @override
    def get_current_value(self) -> float:
        """Return the value, minus any mortgage."""
        return self.value - self.mortgage_balance

    @override
    def deposit(self, amount: float):
        """Pay down the mortgage balance. If overpaid, reduce mortgage to zero and add excess to property value."""
        if amount <= 0:
            return

        self.mortgage_balance -= amount
        if self.mortgage_balance < 0:
            excess = -self.mortgage_balance
            self.mortgage_balance = 0
            self.value += excess
            logger.debug("Mortgage paid off, excess %.2f added to property value", excess)

    @override
    def withdraw(self, amount: float) -> float:
        """Increase mortgage balance (cash-out refinance) up to available equity."""
        if amount <= 0:
            return 0.0

        equity = self.value - self.mortgage_balance
        if equity <= 0:
            return 0.0

        withdraw_amount = min(amount, equity)
        self.mortgage_balance += withdraw_amount
        logger.debug("Withdrew %.2f, new mortgage balance %.2f", withdraw_amount, self.mortgage_balance)
        return withdraw_amount
