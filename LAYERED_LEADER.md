# Retained layered qMACE leader: CPU / CUDA inference

This is the `pointinit6_kernel` configuration, subsequently named `mating_current`.
It is an explicit opt-in. The existing tidyHEBO defaults, fitting code, input/output
warping and DataFrame API remain unchanged. No experiment runner is required.

```python
from tidyhebo import tidyHEBO, RealP
from tidyhebo.layered import leader_config

cfg = leader_config(q=4, inference_device="cuda", qmc_seed=42)
opt = tidyHEBO([RealP(0, 0, 1), RealP(1, 0, 1)], n_init=4, config=cfg)
# Repeat until your evaluation budget is exhausted:
X = opt.suggest()               # DataFrame, up to q initialization rows, then q rows
# y = evaluate(X)               # array-like (len(X), 1), MAXIMIZATION
# opt.observe(X, y)             # negate a minimization loss before observe
```

To use existing observations, pass `train_x=frame, train_y=y` instead of `n_init`.
For the experiment seed manager, pass its trajectory seed as `qmc_seed` and set
NumPy/Python/Torch RNGs as before. This library does not introduce its own seed list.
The search does not change Torch's global dtype or the caller's thread settings.

`inference_device` accepts `"cpu"` (default), `"cuda"`, `"cuda:0"`, or `"auto"`.
An explicit unavailable CUDA device raises before fitting, rather than silently
falling back. `auto` chooses CUDA if available, otherwise CPU. Only CPU and CUDA
are supported. All GP/AF and gradient-coordinate computation is float64.

## What the preset contains

- The existing joint GP fit: Matérn 5/2, Kumaraswamy input warp, Yeo–Johnson + gate.
- Point initialization: 48 candidates; anchors for LogEI/PI/UCB; six selected
  directions from the H=6 Das–Dennis grid; three Adam restarts per target,
  30 steps, lr=0.04. No evolutionary point optimization (`initial_only`).
- Point pool: nondominated layers, including deeper layers as needed, cap 96.
- Pool-index mini-GA: population 48, five generations, scores of complete q-batches.
- Joint batch initialization: anchors + six directions, three restarts and 30
  Adam steps, plus the pool-index batches, then Sobol filling to 48 candidates.
- Whole-batch NSGA-III: population 48, offspring count 48, 20 generations;
  qLogEI/qPI/qUCB. Full ordinary mixed-variable crossover/mutation after
  Hungarian parent-point alignment using kernel correlation distance, including
  fitted input warp and ARD. No point_inherit replacement or adaptive directions.
- Final memetic refinement: select distinct directional winners, cap ceil(0.2*N),
  15 Adam steps, merge with the population and run the usual survival selection.
- No population history; uniformly draw a complete batch from the final front.
  `return_all=True` instead returns the whole front with the standard batch/point
  MultiIndex. Observed/within-batch duplicate points are filtered; exhaustion raises.

Scalarization: L_i = 1-(s_i-low_i)/scale_i, with min/range fixed from 256 Sobol
samples, range floor 1e-10 and ideal z_i=-0.05. For ray r, weights
w_i=1/max(r_i,0.02), WITHOUT sum-normalization. Minimize
`max_i(w_i*(L_i-z_i)) + 0.01*sum_i(w_i*(L_i-z_i))`.
This is the nonsmooth augmented Chebyshev leader, not the 91-direction smooth
Chebyshev / no-GA prototype. The point preparation explicitly selects its evaluator:
analytic Gaussian AFs without YJ, quadrature with active YJ. Joint q>1 uses
QMC with 512 samples. Point quadrature size is 128 in the preset.

The opt-in leader supports continuous `RealP`/`RealPowP` domains, including the
existing external stick-breaking representation of simplex problems. It does not
add simplex transforms itself. Other parameter types retain the old searches;
layered search rejects them before fitting. q is not hard-coded to 4.

## What moves to the GPU

A separate frozen GP + acquisition copy is created once per suggest. The fitting
model, train_x/train_y and parameter state_dict stay on CPU. No `.cuda()` call
mutates the fitting model. Input-warp training buffers and YJ scale follow the
copy. No diagnostics dictionary is attached to the optimizer.

All 3 anchor and 6 directional targets are optimized as 27 independent restart
rows of one `[27,q,d]` Adam tensor. The final memetic candidates are also optimized
together. Each row keeps its own best feasible candidate, including its start.
Bounds, novelty, distinctness and scalarization stay on the inference device.
QMC posterior samples/inverse warp are shared between the three criteria only
when the native sampler seeds coincide; different seeds keep their original
native estimators. BoTorch's stable native qLogEI reductions are preserved.

The population AF evaluations also use the inference device. Pymoo's discrete
operators, Hungarian/kernel parent matching, deduplication, nondominated sorting
and public pandas conversion remain CPU work. GPU is therefore NOT promised to
accelerate the entire BO step by the standalone AF speedup. GP fitting is unchanged.

`evaluation_chunk=32` retains the old evaluation chunk. To try larger population
AF batches, use `replace(cfg, search=replace(cfg.search, evaluation_chunk=128))`.
This is an evaluation chunk, not a worker-count setting. Python multiprocessing
is not used inside suggest; multiple processes can contend for one GPU.

The grouped floating-point calculations and removal of the redundant one-element
random selection mean numerical trajectories need not be bitwise identical to
historical experiment runs. The architecture, AF estimators, budgets and independent
restart objectives are preserved. This port has NOT been re-benchmarked on Olympus.

## Install and check CUDA

Keep the pinned Torch 2.7.1 / BoTorch 0.14.0 versions. Install the CUDA wheel FIRST
in the intended environment, not into the old TensorFlow/Olympus environment:

```powershell
python -m pip install --force-reinstall torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests/test_layered_inference.py tests/test_layered_warped.py -q
```

The CUDA tests check values AND coordinate gradients against CPU float64 and run
actual layered suggestions, including YJ/input warp. If CUDA is unavailable they
are skipped. Install pytest separately or use the repository's test extra.
Torch wheels/driver compatibility: https://pytorch.org/get-started/previous-versions/

## Architecture and scalarization

- [Method and module responsibilities](docs/layered-search.md)
- [Augmented Chebyshev formulas](docs/chebyshev-scalarization.md)
- [Editable method diagram](docs/diagrams/layered-search.drawio)
- [Refactoring validation](docs/validation/layered-refactor.md)

CPU/CUDA performance measurement belongs in the benchmark repository. The
standalone speed probe and its optimizer search_state dictionary were removed
from this library. Device placement and correctness tests remain supported.
