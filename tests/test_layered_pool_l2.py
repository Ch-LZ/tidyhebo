"""Seeded old-vs-new L2 replay using a deterministic pymoo double.

These tests isolate the structural move from numerical behavior of the
optimizer, and exercise the Sobol pool-extension branch as well.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pymoo.core.problem import Problem

from tidyhebo.layered import pool_batches_l2
from tidyhebo.layered.search import LayeredSearch


class _Population:
    def __init__(self, x):
        self.x = x

    def get(self, key):
        assert key == "X"
        return self.x


class _Algorithm:
    """Runs a mocked generation with real problem evaluation and RNG draws."""

    def __init__(self, *, pop_size, sampling, ref_dirs=None):
        self.pop = _Population(sampling)
        self.pop_size = pop_size
        self.ref_dirs = ref_dirs

    def setup(self, problem, *, termination, seed):
        self.problem = problem
        self.remaining = termination[1]
        self.seed = seed
        assert termination[0] == "n_gen"
        return self

    def has_next(self):
        return self.remaining > 0

    def next(self):
        out = {}
        self.problem._evaluate(self.pop.get("X"), out)
        assert out["F"].shape == (self.pop_size, self.problem.n_obj)
        assert out["G"].shape == (self.pop_size, 1)
        self.pop.x = self.pop.x[np.random.permutation(self.pop_size)]
        self.remaining -= 1


def _legacy_small_batches_ga(search, point_pool):
    """Original method, frozen for the refactor regression test."""
    init_config = search.config.batch_init
    pool = point_pool.squeeze(1)
    if len(pool) < search.q:
        search.log(f"point pool has {len(pool)} < q={search.q}; extending with Sobol points")
        extra = search.sobol(search.q * 2).reshape(-1, search.d)
        pool = torch.unique(torch.cat((pool, extra)), dim=0)
    q, n = search.q, search.level.population

    class LegacyPoolProblem(Problem):
        def __init__(self):
            super().__init__(n_var=q, n_obj=search.acq.n_obj, xl=0, xu=len(pool) - 1, n_ieq_constr=1)

        def _evaluate(self, x, out, **kwargs):
            idx = np.rint(x).astype(int)
            batches = pool[torch.as_tensor(idx, device=pool.device)]
            with torch.no_grad():
                out["F"] = -search.scores(batches).cpu().numpy()
            distinct = np.array([len(set(row)) == q for row in idx])
            out["G"] = (0.5 - (distinct & search.feasible(batches).cpu().numpy()))[:, None]

    initial = np.stack([np.random.choice(len(pool), q, replace=False) for _ in range(n)])
    algorithm = (
        _Algorithm(pop_size=n, sampling=initial)
        if search.acq.n_obj == 1
        else _Algorithm(pop_size=n, sampling=initial, ref_dirs=search.directions)
    )
    algorithm.setup(
        LegacyPoolProblem(),
        termination=("n_gen", init_config.pool_generations),
        seed=int(np.random.randint(2**31 - 1)),
    )
    while algorithm.has_next():
        algorithm.next()
    idx = np.rint(algorithm.pop.get("X")).astype(int)
    return search.clean(pool[torch.as_tensor(idx, device=pool.device)])


class _SearchDouble:
    def __init__(self, q, n_obj):
        self.q, self.d = q, 2
        self.acq = SimpleNamespace(n_obj=n_obj)
        self.level = SimpleNamespace(population=8)
        self.config = SimpleNamespace(batch_init=SimpleNamespace(pool_generations=3))
        self.directions = np.ones((4, n_obj))
        self.messages = []

    def log(self, msg):
        self.messages.append(msg)

    def sobol(self, n):
        return torch.rand((n, self.q, self.d), dtype=torch.double)

    def scores(self, batches):
        s = batches.sum(dim=(-1, -2))
        return torch.stack([(s + 1) * (i + 1) for i in range(self.acq.n_obj)], -1)

    def feasible(self, batches):
        return torch.isfinite(batches).all(dim=(-1, -2))

    def clean(self, batches):
        return batches.clone()


def _np_state_equal(left, right):
    return all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
               for x, y in zip(left, right))


@pytest.mark.parametrize("q,n_obj,pool_size", [(1, 1, 6), (2, 3, 6), (4, 3, 2)])
def test_l2_matches_previous_result_and_rng(monkeypatch, q, n_obj, pool_size):
    monkeypatch.setattr(pool_batches_l2, "GA", _Algorithm)
    monkeypatch.setattr(pool_batches_l2, "NSGA3", _Algorithm)
    point_pool = torch.arange(1, pool_size * 2 + 1, dtype=torch.double).reshape(pool_size, 1, 2) / 100

    def run(method):
        np.random.seed(100 + q)
        torch.manual_seed(200 + q)
        search = _SearchDouble(q, n_obj)
        result = method(search, point_pool)
        return result, np.random.get_state(), torch.random.get_rng_state(), search.messages

    before = run(_legacy_small_batches_ga)
    after = run(LayeredSearch.small_batches_ga)
    assert torch.equal(before[0], after[0])
    assert _np_state_equal(before[1], after[1])
    assert torch.equal(before[2], after[2])
    assert before[3] == after[3]
