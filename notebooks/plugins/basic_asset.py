from .base_asset import BaseAsset
import numpy as np

class BasicAsset(BaseAsset):
    """
    An asset with a specified return and volatility.
    Perfect for stablecoins, simple real estate equity, etc.
    """
    def __init__(self, params: dict):
        super().__init__(params)
        self.expected_return = params['expected_return']
        self.volatility = params['volatility']

    def process_annual_step(self):
        if self.value > 0:
            growth = np.random.normal(self.expected_return, self.volatility)
            self.value *= (1 + growth)
