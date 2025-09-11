from pydantic import BaseModel

# Number of trading days in a year (used for daily returns)
num_trading_days: int = 252


class BasicAssetParams(BaseModel):
    initial_investment: float
    expected_return: float
    volatility: float


class StockPortfolioParams(BaseModel):
    initial_investment: float
    portfolio_mix: dict[str, float]


class AssetConfig(BaseModel):
    type: str
    params: BasicAssetParams | StockPortfolioParams


class AssetAction(BaseModel):
    type: str
    name: str
    cost: float | None = None
    funding_priority: list[str] | None = None
    config: AssetConfig | None = None


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
