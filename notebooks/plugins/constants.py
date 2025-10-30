from typing import Literal
from pydantic import BaseModel, field_validator, model_validator, Field

# Number of trading days in a year (used for daily returns)
num_trading_days: int = 252


class BasicAssetParams(BaseModel):
    initial_investment: float
    expected_return: float
    volatility: float
    process_frequency: Literal["daily", "monthly", "yearly"] = "monthly"


class StockPortfolioParams(BaseModel):
    initial_investment: float
    portfolio_mix: dict[str, float]


class RealEstateParams(BaseModel):
    property_value: float
    down_payment: float
    mortgage_rate: float
    mortgage_term_years: int
    expected_return: float
    volatility: float


class AssetConfig(BaseModel):
    type: Literal["basic_asset", "stock_portfolio", "mortgaged_real_estate"]
    params: BasicAssetParams | StockPortfolioParams | RealEstateParams

    @field_validator("params", mode="before")
    @classmethod
    def validate_params(cls, v, info):
        asset_type = info.data.get("type")
        if asset_type == "basic_asset":
            BasicAssetParams(**v)
        elif asset_type == "stock_portfolio":
            StockPortfolioParams(**v)
        elif asset_type == "mortgaged_real_estate":
            RealEstateParams(**v)
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")
        return v



class GrantAssetAction(BaseModel):
    """Action to grant (add) an asset without requiring funding."""
    name: str
    config: AssetConfig


class BuyAssetAction(BaseModel):
    """Action to purchase an asset by withdrawing funds from other assets."""
    name: str
    cost: float = Field(gt=0, description="Cost must be greater than 0. Use grant_asset if cost is 0.")
    funding_priority: list[str] = Field(min_length=1, description="Must specify at least one funding source")
    config: AssetConfig


class SellAssetAction(BaseModel):
    """Action to sell (liquidate) an asset and transfer proceeds to another asset."""
    name: str
    destination: str


class ModifyAssetAction(BaseModel):
    """Action to modify parameters of an existing asset."""
    name: str
    updates: dict


class AssetAction(BaseModel):
    type: Literal["grant_asset", "buy_asset", "sell_asset", "modify_asset"]
    name: str
    cost: float | None = None
    funding_priority: list[str] | None = None
    config: AssetConfig | None = None
    destination: str | None = None  # For sell_asset
    updates: dict | None = None  # For modify_asset

    @model_validator(mode="after")
    def validate_action_fields(self):
        """Validate that required fields are present based on action type."""
        action_type = self.type

        # Validate required fields for each action type by attempting to construct the specific action model
        if action_type == "grant_asset":
            GrantAssetAction(name=self.name, config=self.config)
        elif action_type == "buy_asset":
            BuyAssetAction(
                name=self.name,
                cost=self.cost,
                funding_priority=self.funding_priority,
                config=self.config
            )
        elif action_type == "sell_asset":
            SellAssetAction(name=self.name, destination=self.destination)
        elif action_type == "modify_asset":
            ModifyAssetAction(name=self.name, updates=self.updates)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        return self


class LifePhase(BaseModel):
    name: str
    age: int
    # Annual income during this phase
    annual_income: float | None = None
    # Non discretionary expenses every year - if not covered by annual_income, will be withdrawn from assets
    annual_expenses: float | None = None
    annual_investment: float | None = None
    withdraw_order: list[str] | None = None
    investment_allocation: dict[str, float] | None = None
    actions: list[AssetAction] | None = None


class LifePhases(BaseModel):
    life_phases: list[LifePhase]
