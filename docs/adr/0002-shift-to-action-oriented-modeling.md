# 2. Shift to action-oriented modeling

Date: 2025-09-10

## Status

Accepted

## Context

The original simulation design required a static, upfront `asset_classes_config` to instantiate and carry a fixed list of assets through the entire run. The asset-first modeling doesn't allow for life changes such as starting with no assets, buying and selling houses or other assets, or changing investment strategies over time.

## Decision

Refactor the simulation to be action-oriented:

- `run_multi_asset_simulation` now requires only `life_phases_config` (plus the lifecycle/flow functions). It no longer accepts or depends on a global `asset_classes_config`.
- Assets are introduced, modified, or removed over time via `actions` embedded in the `life_phases` configuration (e.g., `buy_asset`, `sell_asset`, `modify_asset`).
- Asset instantiation is performed progressively as the simulation advances. When an action introduces a new asset, the system uses the configured type and params to initialize it, keeping the asset registry in sync with the scenario timeline.

## Consequences

- Configuration becomes more expressive: scenarios describe “what happens when” (actions) rather than frontloading all assets.
