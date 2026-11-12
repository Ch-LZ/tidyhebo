"""L2: select complete q-point batches from the L1 Pareto point pool.

The pool is a fixed tensor of normalized GP points, shape ``(N, d)``.
Each pymoo individual is a length-q tuple of *pool indices*, not q real
coordinates. The search maximizes acquisition objectives subject to distinct
indices and a caller-provided feasibility predicate. The resulting tensor has
shape ``(population, q, d)``; the caller cleans/canonicalizes it afterward.

The only external dependencies of this layer are its acquisition evaluator and
feasibility predicate. Neither LayeredSearch nor its config tree is required.
"""

from collections.abc import Callable

import numpy as np
import torch
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem


class PoolProblem(Problem):
    """Minimization adapter for pymoo over *indices into a fixed point pool*.

    ``pool`` has shape ``(N, d)``. ``q`` is the number of points per batch,
    and ``n_obj`` is the number of acquisition objectives (one or more).

    ``acquisition_scores(batches)`` evaluates normalized GP-coordinate batches
    of shape ``(M, q, d)`` and returns ``(M, n_obj)`` objective values to
    **maximize**. Pymoo minimizes, hence ``F = -acquisition_scores``.

    ``feasible(batches)`` returns a boolean mask of shape ``(M,)``. In addition
    to this predicate, repeated pool indices within one batch are forbidden.
    A feasible individual has ``G = -0.5``; an infeasible one has ``G = 0.5``.
    The pool and callbacks are fixed throughout one GA/NSGA-III run.
    """

    def __init__(
        self,
        pool: torch.Tensor,
        *,
        q: int,
        n_obj: int,
        acquisition_scores: Callable[[torch.Tensor], torch.Tensor],
        feasible: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__(n_var=q, n_obj=n_obj, xl=0, xu=len(pool) - 1, n_ieq_constr=1)
        self.pool = pool
        self.q = q
        self.acquisition_scores = acquisition_scores
        self.feasible = feasible

    def _evaluate(self, x: np.ndarray, out: dict, **kwargs) -> None:
        """Map rounded pool-index rows to pymoo objectives and constraints."""
        idx = np.rint(x).astype(int)
        batches = self.pool[torch.as_tensor(idx, device=self.pool.device)]
        with torch.no_grad():
            out["F"] = -self.acquisition_scores(batches).cpu().numpy()
        distinct = np.array([len(set(row)) == self.q for row in idx])
        out["G"] = (0.5 - (distinct & self.feasible(batches).cpu().numpy()))[:, None]


def solve_pool_problem(
    problem: PoolProblem,
    *,
    population: int,
    generations: int,
    ref_dirs: np.ndarray,
) -> torch.Tensor:
    """Evolve q-tuples of point indices and return decoded GP-coordinate batches.

    ``problem`` provides the pool and its AF/feasibility evaluation. The
    population contains ``population`` index tuples sampled **without**
    replacement; the search runs for ``generations`` pymoo generations.
    A single objective uses GA, multiple objectives use NSGA-III with the
    supplied ``ref_dirs`` (shape ``(R, n_obj)``).

    Returns a tensor ``(population, q, d)`` on the same device/dtype as the
    original pool. This function intentionally does *not* canonicalize or
    deduplicate the returned batches; the caller performs its normal cleanup.

    RNG order is fixed: NumPy population choices, algorithm construction,
    NumPy draw for pymoo's seed, then pymoo generations. Do not add reseeding
    or extra draws here: existing benchmark trajectories rely on this order.
    """
    initial = np.stack([
        np.random.choice(len(problem.pool), problem.q, replace=False)
        for _ in range(population)
    ])
    algorithm = (
        GA(pop_size=population, sampling=initial)
        if problem.n_obj == 1
        else NSGA3(pop_size=population, sampling=initial, ref_dirs=ref_dirs)
    )
    algorithm.setup(
        problem,
        termination=("n_gen", generations),
        seed=int(np.random.randint(2**31 - 1)),
    )
    while algorithm.has_next():
        algorithm.next()
    idx = np.rint(algorithm.pop.get("X")).astype(int)
    return problem.pool[torch.as_tensor(idx, device=problem.pool.device)]
