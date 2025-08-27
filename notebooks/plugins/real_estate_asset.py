from .base_asset import BaseAsset
import numpy as np
import numpy_financial as npf

class MortgagedRealEstateAsset(BaseAsset):
    """
    Represents a real estate property with a mortgage.
    Tracks property value appreciation and mortgage paydown.
    'value' in this context represents the owner's equity.
    """
    def __init__(self, params: dict):
        # We don't call super().__init__ because initial_investment is equity, not value.
        self.params = params
        self.property_value = params['property_value']
        self.mortgage_balance = self.property_value - params['down_payment']
        self.rate = params['mortgage_rate']
        self.term = params['mortgage_term_years']

        # Calculate monthly payment for P&I
        if self.mortgage_balance > 0:
            monthly_rate = self.rate / 12
            nper = self.term * 12
            self.monthly_payment = -npf.pmt(monthly_rate, nper, self.mortgage_balance)
            self.annual_payment = self.monthly_payment * 12
        else:
            self.monthly_payment = 0
            self.annual_payment = 0

        self.appreciation_rate = params['appreciation_rate']
        self.appreciation_volatility = params['appreciation_volatility']

        # Update the public `value` to be the initial equity
        self.value = self.get_value()

    def process_annual_step(self):
        # 1. Appreciate the property value
        growth = np.random.normal(self.appreciation_rate, self.appreciation_volatility)
        self.property_value *= (1 + growth)

        # 2. Pay down the mortgage for one year
        if self.mortgage_balance > 0:
            # Simple annual calculation for simulation purposes
            interest_paid = self.mortgage_balance * self.rate
            principal_paid = self.annual_payment - interest_paid
            self.mortgage_balance -= principal_paid
            self.mortgage_balance = max(0, self.mortgage_balance) # Don't go below zero

        # 3. Update the asset's value (equity)
        self.value = self.get_value()

    def get_value(self) -> float:
        """For this asset, value is equity."""
        return self.property_value - self.mortgage_balance

    def get_annual_income(self) -> float:
        """Calculate net operating income (cash flow)."""
        gross_rent = self.params.get('annual_gross_rent', 0)
        expenses = gross_rent * self.params.get('annual_expenses_percent', 0)
        net_income = gross_rent - expenses - self.annual_payment
        return net_income

    def withdraw(self, amount: float) -> float:
        # Withdrawal from a mortgaged property is complex (HELOC, sale).
        # For now, we'll assume it's illiquid and cannot be withdrawn from.
        print(f"WARNING: Withdrawal from '{self.params.get('name', 'mortgaged_real_estate')}' not supported. Asset is illiquid.")
        return 0.0
