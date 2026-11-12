# Project diagrams

Open `project-structure.drawio` in diagrams.net or VS Code Draw.io Integration.
It contains five editable pages:

1. Classes.
2. Classes with A0–A11 ablation markers and the legend below.
3. Classes with S1–S6 speed markers and the legend below.
4. Current configuration classes, typed fields and defaults.
5. Layered method: point preparation, pool-index GA, full-batch GA and refinement.

Maintain all pages manually in the same commit as related code changes.
Keep class structure consistent across pages 1–3. Update config fields/defaults
on page 4 and the relevant A/S labels when changing a setting or implementation.
READY, PARTIAL, PENDING, EXCLUDED and DEFERRED distinguish current code from plans.

## Finding the gate

A6 is `ModelConfig.gate`, default True. `SimpleModel.train_model()` checks it and
can refit without YJ when std(train_targets) < 0.5. Both other model classes
inherit that workflow. There is no `use_gate` config field now.

To disable it for an existing optimizer with frozen configs:

```python
from dataclasses import replace
optimizer.config = replace(
    optimizer.config, model=replace(optimizer.config.model, gate=False),
)
```

`mll-hierarchy.drawio` separately shows BaseMLL and its injected GP calculations.
Keep drawio XML compression disabled for readable Git diffs.

`layered-search.drawio` is the standalone editable method diagram. Its SVG preview
is `previews/layered-search.svg`. The flow and default budgets correspond to
`LayeredSearch.search` and `leader_config`; see `../layered-search.md` for details.
