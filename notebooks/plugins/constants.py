from pydantic import BaseModel

# Number of trading days in a year (used for daily returns)
num_trading_days = 252


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
    cost: float
    funding_priority: list[str] | None
    config: AssetConfig


class LifePhase(BaseModel):
    name: str
    age: int
    annual_income: float
    annual_expenses: float
    annual_investment: float
    investment_allocation: dict[str, float]
    actions: list[AssetAction] | None


class LifeModelConfig(BaseModel):
    description: str | None
    num_simulations: int | None
    life_phases: list[LifePhase]
