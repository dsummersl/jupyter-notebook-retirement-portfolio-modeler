from .base_asset import BaseAsset
import logging

logger = logging.getLogger(__name__)


def buy_asset(
    assets: dict, params: dict, asset_map: dict
) -> tuple[list[str], list[str]]:
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
    cost = params["cost"]
    funding_priority = params["funding_priority"]
    asset_name = params["name"]
    asset_config = params["config"]

    if asset_name in assets:
        logger.warning(
            f"Warning: Asset '{asset_name}' already exists. Skipping buy action."
        )
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
        asset_config["params"]["initial_investment"] = withdrawn_amount
        asset_config["params"]["down_payment"] = (
            withdrawn_amount  # For mortgaged assets
        )
        asset_config["params"]["name"] = asset_name

        asset_class = asset_map.get(asset_config["type"])
        if not asset_class:
            raise ValueError(
                f"Unknown asset type '{asset_config['type']}' in buy_asset action."
            )

        new_asset = asset_class(asset_config["params"])
        assets[asset_name] = new_asset
        return [asset_name], []

    return [], []


def sell_asset(
    assets: dict, params: dict, asset_map: dict
) -> tuple[list[str], list[str]]:
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
        logger.warning(
            f"Warning: Cannot sell asset '{asset_name}' because it does not exist."
        )
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


def modify_asset(
    assets: dict, params: dict, asset_map: dict
) -> tuple[list[str], list[str]]:
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
        logger.warning(
            f"Warning: Cannot modify asset '{asset_name}' as it does not exist."
        )
        return [], []

    asset_to_modify = assets[asset_name]
    # Delegate the modification logic to the asset object itself
    asset_to_modify.modify(updates)

    return [], []


# This map is the bridge between the configuration string and the function to execute
ACTION_HANDLER_MAP = {
    "buy_asset": buy_asset,
    "sell_asset": sell_asset,
    "modify_asset": modify_asset,
}
