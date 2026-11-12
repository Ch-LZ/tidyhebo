# tidyHEBO

Bayesian optimization with jointly fitted Gaussian-process/output warping and
the qMACE acquisition ensemble. The optimizer maximizes one scalar objective;
negate a loss when minimizing. Sequential and batch suggestions use the same
DataFrame `suggest()` / `observe()` interface.

## Install

tidyHEBO supports Python 3.10 through 3.14. From a checkout, install it with
pip in your Python environment, including an active Mamba environment:

```powershell
python -m pip install .
```

The package declares compatible dependency ranges so the resolver can use
versions that also work with your other packages. To select a specific CPU or
CUDA build, follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/)
before installing tidyHEBO.
For paper reproduction, use the frozen environment from the separate
[tidyhebo_results repository](https://github.com/Ch-LZ/tidyhebo_results).

## Minimal example

```python
import random
import numpy as np
import torch
from tidyhebo import tidyHEBO, RealP, leader_config

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)

optimizer = tidyHEBO(
    [RealP(0, 0.0, 1.0, name="x")], n_init=4,
    config=leader_config(q=1, qmc_seed=42),
)
for _ in range(6):
    X = optimizer.suggest()
    y = -((X[["x"]].to_numpy() - 0.35) ** 2)
    optimizer.observe(X, y)
```

`suggest()` returns a pandas DataFrame. Supply the same parameter columns to
`observe()` and objective values of shape `(n, 1)`. `observe()` appends data;
`suggest()` fits the model and searches for candidates. For a batch, select
`leader_config(q=4, qmc_seed=42)`; the returned frame then contains four rows.

The explicit `leader_config` selects the retained layered search used by the
paper. The generic constructor also supports other searches and does not
implicitly select a paper experiment configuration.

## Paper presets

```python
from tidyhebo import RealP, tidyHEBO
from tidyhebo.presets import tidyHEBO_QMACE

optimizer = tidyHEBO.from_config(tidyHEBO_QMACE, [RealP(0, 0.0, 1.0, name="x")], n_init=4)
```

`tidyHEBO_QLOGEI` and `VANILLA_GP_QLOGEI` provide the warped and unwarped
GP comparisons.
Batch presets default to four points. Use `dataclasses.replace` to select
another batch size or set the acquisition seed. See the
[paper presets notebook](examples/02_preset_configs.ipynb) for a complete example.
The [hyperparameter example](examples/01_example.ipynb) shows a mixed search space.

## Checks and documentation

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

- [Layered candidate search](docs/layered-search.md)
- [Search scalarization](docs/chebyshev-scalarization.md)
- [Paper experiments and historical results](https://github.com/Ch-LZ/tidyhebo_results)

Paper reproduction requires its frozen seeds, initial designs, task assets and
campaign settings. A seed alone does not guarantee bitwise equality across
Python, dependency versions, operating systems and numerical backends.

## Cite

Citation details will be added here upon publication.

## License

The existing project declaration is CC BY-NC 4.0; see [LICENSE.md](LICENSE.md).
