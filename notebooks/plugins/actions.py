import logging
from typing import Any

from .constants import AssetAction

logger = logging.getLogger(__name__)


def buy_asset(assets: dict, params: AssetAction | dict, asset_map: dict[str, type]) -> tuple[list[str], list[str]]:
    """
    Handles the purchase of a new asset mid-simulation.

    Args:
        assets (dict): The current dictionary of asset objects in the simulation.
        params (dict): The action's parameters from the life_phases config.
                       Expected keys: 'name', 'cost', 'funding_priority', 'config'.
        asset_map (dict): The global mapping from asset type strings to classes.

    Returns:
        A tuple of (assets_added, assets_removed).
    """
    # Support both dict-config (from YAML) and pydantic AssetAction
    if isinstance(params, AssetAction):
        asset_name = params.name
        cost = float(getattr(params, "cost", 0) or 0)
        funding_priority = list(params.funding_priority or [])
        cfg = params.config
        # pydantic v2 model_dump() will descend into nested models
        asset_config: dict[str, Any] = cfg.model_dump()
    else:
        cost = params.get("cost", 0)
        funding_priority = params.get("funding_priority", [])
        asset_name = params["name"]
        asset_config = params["config"]

    if asset_name in assets:
        logger.warning(f"Warning: Asset '{asset_name}' already exists. Skipping buy action.")
        return [], []

    # Fund the purchase by withdrawing from specified assets
    withdrawn_amount = 0
    for source_asset_name in funding_priority:
        amount_to_withdraw = cost - withdrawn_amount
        if amount_to_withdraw <= 0:
            break

        if source_asset_name in assets:
            actual_withdrawn = assets[source_asset_name].withdraw(amount_to_withdraw)
            withdrawn_amount += actual_withdrawn

    if withdrawn_amount < cost:
        logger.warning(
            f"Warning: Insufficient funds for '{asset_name}'. "
            f"Needed ${cost:,.0f}, got ${withdrawn_amount:,.0f}."
        )

    # Create and add the new asset if any funds were secured
    if withdrawn_amount > 0:
        # Inject the actual funds into the asset's parameters
        # For a basic asset, this is 'initial_investment'. For a house, 'down_payment'.
        asset_config.setdefault("params", {})
        asset_config["params"]["initial_investment"] = withdrawn_amount
        asset_config["params"]["down_payment"] = withdrawn_amount  # For mortgaged assets
        asset_config["params"]["name"] = asset_name

        # Use initialize_assets to construct the asset to keep behavior consistent
        from .modeler import initialize_assets  # local import to avoid circular at module load

        new_assets = initialize_assets({asset_name: asset_config})
        assets[asset_name] = new_assets[asset_name]
        return [asset_name], []

    return [], []


def sell_asset(assets: dict, params: dict, asset_map: dict) -> tuple[list[str], list[str]]:
    """
    Handles the sale (liquidation) of an existing asset.

    Args:
        assets (dict): The current dictionary of asset objects.
        params (dict): Expected keys: 'name' (asset to sell),
                       'destination' (asset to deposit proceeds into).
        asset_map (dict): Not used here, but kept for consistent signature.

    Returns:
        A tuple of (assets_added, assets_removed).
    """
    asset_name = params["name"]
    destination_name = params["destination"]

    if asset_name not in assets:
        logger.warning(f"Warning: Cannot sell asset '{asset_name}' because it does not exist.")
        return [], []

    if destination_name not in assets:
        logger.warning(
            f"Warning: Cannot deposit proceeds into '{destination_name}' because it does not exist."
        )
        # Optionally, you could create a cash asset here. For now, we fail.
        return [], []

    asset_to_sell = assets[asset_name]
    sale_proceeds = asset_to_sell.get_current_value()

    # You could add a 'transaction_cost_percent' to params for more realism
    # sale_proceeds *= (1 - params.get('transaction_cost_percent', 0.0))

    # Add proceeds to destination and remove the sold asset
    assets[destination_name].value += sale_proceeds
    del assets[asset_name]

    logger.info(
        f"Sold '{asset_name}' for ${sale_proceeds:,.0f}, proceeds moved to '{destination_name}'."
    )
    return [], [asset_name]


def modify_asset(assets: dict, params: dict, asset_map: dict) -> tuple[list[str], list[str]]:
    """
    Modifies the parameters of an existing asset.

    Args:
        assets (dict): The current dictionary of asset objects.
        params (dict): Expected keys: 'name' (asset to modify),
                       'updates' (dict of parameters to change).
        asset_map (dict): Not used here.

    Returns:
        A tuple of (assets_added, assets_removed).
    """
    asset_name = params["name"]
    updates = params["updates"]

    if asset_name not in assets:
        logger.warning(f"Warning: Cannot modify asset '{asset_name}' as it does not exist.")
        return [], []

    asset_to_modify = assets[asset_name]
    # Delegate the modification logic to the asset object itself
    asset_to_modify.modify(updates)

    return [asset_name], [asset_name]


def grant_asset(assets: dict, params: AssetAction | dict, asset_map: dict[str, type]) -> tuple[list[str], list[str]]:
    """
    Grants (adds) an asset to the simulation without requiring funding/withdrawals.

    Args:
        assets (dict): The current dictionary of asset objects in the simulation.
        params (AssetAction | dict): Expected keys/fields: 'name', 'config'.
        asset_map (dict[str, type]): The global mapping from asset type strings to classes.

    Returns:
        A tuple of (assets_added, assets_removed).
    """
    # Normalize params similar to buy_asset
    if isinstance(params, AssetAction):
        asset_name = params.name
        cfg = params.config
        asset_config: dict[str, Any] = cfg.model_dump()
    else:
        asset_name = params["name"]
        asset_config = params["config"]

    if asset_name in assets:
        logger.warning(f"Warning: Asset '{asset_name}' already exists. Skipping grant action.")
        return [], []

    # Ensure required params exist and inject the name
    asset_config.setdefault("params", {})
    asset_config["params"]["name"] = asset_name
    asset_config["params"].setdefault("initial_investment", 0)

    # Construct and register the asset
    from .modeler import initialize_assets  # local import to avoid circular at module load

    new_assets = initialize_assets({asset_name: asset_config})
    assets[asset_name] = new_assets[asset_name]
    logger.info(f"Granted asset '{asset_name}' with config type='{asset_config.get('type')}'.")
    return [asset_name], []

# This map is the bridge between the configuration string and the function to execute
ACTION_HANDLER_MAP = {
    "buy_asset": buy_asset,
    "sell_asset": sell_asset,
    "modify_asset": modify_asset,
    "grant_asset": grant_asset,
}
