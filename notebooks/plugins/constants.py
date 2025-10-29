from typing import Literal
from pydantic import BaseModel, field_validator

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



class AssetAction(BaseModel):
    type: Literal["grant_asset", "buy_asset", "sell_asset", "modify_asset"]
    name: str
    cost: float | None = None
    funding_priority: list[str] | None = None
    config: AssetConfig | None = None
    destination: str | None = None  # For sell_asset
    updates: dict | None = None  # For modify_asset


class LifePhase(BaseModel):
    name: str
    age: int
    annual_income: float | None = None
    annual_expenses: float | None = None
    annual_investment: float | None = None
    withdraw_order: list[str] | None = None
    investment_allocation: dict[str, float] | None = None
    actions: list[AssetAction] | None = None


class LifePhases(BaseModel):
    life_phases: list[LifePhase]
