from .base_asset import BaseAsset
import numpy as np


def calculate_monthly_payment(principal, monthly_rate, nper):
    # Formula: P = (r*PV) / (1 - (1 + r)**-n)
    if monthly_rate == 0:
        return principal / nper
    return (monthly_rate * principal) / (1 - (1 + monthly_rate) ** -nper)


class MortgagedRealEstateAsset(BaseAsset):
    """
    Represents a real estate property with a mortgage.
    Tracks property value appreciation and mortgage paydown.
    'value' in this context represents the owner's equity.
    """

    def __init__(self, params: dict):
        # TODO use the pydantic vars
        super().__init__(params)
        self.property_value = params["property_value"]
        self.mortgage_balance = self.property_value - params["down_payment"]
        self.rate = params["mortgage_rate"]
        self.term = params["mortgage_term_years"]

        # Calculate monthly payment for P&I
        if self.mortgage_balance > 0:
            monthly_rate = self.rate / 12
            nper = self.term * 12
            self.monthly_payment = calculate_monthly_payment(
                self.mortgage_balance, monthly_rate, nper
            )
            self.annual_payment = self.monthly_payment * 12
        else:
            self.monthly_payment = 0
            self.annual_payment = 0

        self.asset_params.expected_return = params["expected_return"]
        self.asset_params.volatility = params["volatility"]

        # Update the public `value` to be the initial equity
        self.value = self.get_value()

    # def process_annual_step(self):
    #     # 1. Appreciate the property value
    #     growth = np.random.normal(self.expected_return, self.volatility)
    #     self.property_value *= 1 + growth
    #
    #     # 2. Pay down the mortgage for one year
    #     if self.mortgage_balance > 0:
    #         # Simple annual calculation for simulation purposes
    #         interest_paid = self.mortgage_balance * self.rate
    #         principal_paid = self.annual_payment - interest_paid
    #         self.mortgage_balance -= principal_paid
    #         self.mortgage_balance = max(0, self.mortgage_balance)  # Don't go below zero
    #
    #     # 3. Update the asset's value (equity)
    #     self.value = self.get_value()
    #
    def get_value(self) -> float:
        """For this asset, value is equity."""
        return self.property_value - self.mortgage_balance

    # def get_annual_income(self) -> float:
    #     """Calculate net operating income (cash flow)."""
    #     gross_rent = self.params.get("annual_gross_rent", 0)
    #     expenses = gross_rent * self.params.get("annual_expenses_percent", 0)
    #     net_income = gross_rent - expenses - self.annual_payment
    #     return net_income
    #
    def withdraw(self, amount: float) -> float:
        # Allow borrowing against equity, but warn that this is not an ideal modeling scenario.
        # Consider adding a life phase where the asset is explicitly sold.
        if amount <= 0:
            return 0.0

        equity = self.get_value()
        if equity <= 0:
            # TODO need some kind of logging here that can be captured in the main app to summarize
            # print(
            #     f"WARNING: Withdrawal from '{self.params.get('name', 'mortgaged_real_estate')}' not supported. Asset is illiquid and has no equity."
            # )
            return 0.0

        borrowable = min(float(amount), float(equity))

        # Increase mortgage balance by the borrowed amount (reduces equity)
        self.mortgage_balance += borrowable

        # Recompute payment schedule with updated balance (simplified: uses original full term)
        if self.mortgage_balance > 0:
            monthly_rate = self.rate / 12
            nper = self.term * 12
            self.monthly_payment = calculate_monthly_payment(
                self.mortgage_balance, monthly_rate, nper
            )
            self.annual_payment = self.monthly_payment * 12
        else:
            self.monthly_payment = 0
            self.annual_payment = 0

        # Update exposed value (equity)
        self.value = self.get_value()

        # TODO logging fix
        # print(
        #     f"WARNING: Withdrawal from '{self.params.get('name', 'mortgaged_real_estate')}' executed as borrowing against equity. "
        #     f"This is not ideal for modeling; consider adding a sell_asset action in a life phase."
        # )
        return borrowable
