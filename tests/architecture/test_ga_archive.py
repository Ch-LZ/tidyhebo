from types import SimpleNamespace

import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from pymoo.core.population import Population

import tidyhebo.evolution_optimizer as module
from tidyhebo.archive import ParetoAccumulator
from tidyhebo.config import GASearchConfig
from tidyhebo.design_space import MixedDesignSpace
from tidyhebo.mixed_params import IntegerP, RealP


def front_rows(X, F):
    keep = is_non_dominated(-torch.as_tensor(F), deduplicate=False).numpy()
    return {tuple(row) for row in np.concatenate((X, F), axis=1)[keep]}


def test_callback_uses_feasibility_parameter_order_and_a_fresh_archive(monkeypatch):
    archives = []

    class RecordingArchive(ParetoAccumulator):
        def __init__(self):
            super().__init__()
            archives.append(self)

    monkeypatch.setattr(module, "ParetoAccumulator", RecordingArchive)
    space = MixedDesignSpace([
        RealP(0, name="a", lb=0, ub=1), RealP(1, name="b", lb=0, ub=1),
    ])

    def minimize(problem, algorithm, termination, **kwargs):
        assert kwargs["save_history"] is False
        assert kwargs["copy_algorithm"] is False
        assert kwargs["copy_termination"] is False
        for values, objectives, constraints in [
            ([.1, .2, .3, .4], [[1, 2], [2, 1], [0, 0], [3, 3]], [-1, -1, 1, -1]),
            ([.5, .6, .5], [[.5, 1.5], [3, .5], [.5, 1.5]], [-1, -1, -1]),
        ]:
            population = Population.new(
                "X", np.array([{"b": x / 2, "a": x} for x in values]),
                "F", np.array(objectives, dtype=float),
                "G", np.array(constraints, dtype=float)[:, None],
            )
            kwargs["callback"](SimpleNamespace(pop=population))
        return SimpleNamespace()  # The result comes from the archive, not res.X.

    monkeypatch.setattr(module, "minimize", minimize)
    for _ in range(2):
        points = module.optimize_real_hedged_acquisition(
            SimpleNamespace(n_obj=2), space, torch.empty((0, 2)),
            config=GASearchConfig(pop_size=96, max_generations=2),
        )
        assert {tuple(row) for row in points.tolist()} == {(.2, .1), (.5, .25), (.6, .3)}
        assert not archives[-1].has_data
        assert archives[-1].X is archives[-1].F is None
    assert len(archives) == 2 and archives[0] is not archives[1]


def test_real_pymoo_callbacks_match_full_feasible_history(monkeypatch):
    xs, fs, generations = [], [], []

    class CheckedArchive(ParetoAccumulator):
        def add(self, X, F, feasible=None):
            # Internal GA/archive contract is checked here, not in add().
            assert isinstance(X, np.ndarray) and isinstance(F, np.ndarray)
            assert X.ndim == F.ndim == 2
            assert X.shape[1] == 2 and F.shape[1] == 3
            assert len(X) == len(F)
            assert isinstance(feasible, np.ndarray) and feasible.dtype == np.bool_
            mask = feasible.reshape(-1)
            assert len(mask) == len(X)
            assert np.isfinite(X[mask]).all() and np.isfinite(F[mask]).all()
            xs.append(X[mask].copy()); fs.append(F[mask].copy())
            super().add(X, F, feasible)
            expected = front_rows(np.concatenate(xs), np.concatenate(fs))
            actual = {tuple(row) for row in np.concatenate((self.X, self.F), axis=1)}
            assert actual == expected
            assert len(self.X) == len(expected)
            generations.append(len(self.X))

    class Acquisition(torch.nn.Module):
        n_obj = 3

        def forward(self, X):
            x = X.squeeze(-2)
            return torch.stack((x[..., 0], 1 - x[..., 0], -(x[..., 1] - .5).square()), dim=-1)

    monkeypatch.setattr(module, "ParetoAccumulator", CheckedArchive)
    space = MixedDesignSpace([
        RealP(0, name="real", lb=0, ub=1), IntegerP(1, name="integer", lb=0, ub=3),
    ])
    observed = torch.tensor([[0., 0.]], dtype=torch.double)
    points = module.optimize_real_hedged_acquisition(Acquisition(), space, observed,
        config=GASearchConfig(pop_size=96, max_generations=4))
    assert len(generations) == 4
    assert len(points) > 0
    assert space.are_values_new(points, observed).all()
    assert (points[:, 1] == points[:, 1].round()).all()
