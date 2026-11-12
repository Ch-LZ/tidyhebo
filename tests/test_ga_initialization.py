import numpy as np
import torch
from pymoo.core.problem import Problem

from tidyhebo.design_space import MixedDesignSpace
from tidyhebo.config import GASearchConfig
from tidyhebo.evolution_optimizer import BatchLayout, SobolMixedSampler, optimize_real_hedged_acquisition
from tidyhebo.mixed_params import IntegerP, RealP


def test_scrambled_initialization_changes_between_calls_and_replays_with_torch_seed():
    space = MixedDesignSpace([RealP(0, -2, 3), IntegerP(1, 1, 7)])
    problem = Problem(vars={p.name: p.pymoo_var for p in space.params})
    sampler = SobolMixedSampler(space, BatchLayout(space, q=1))
    with torch.random.fork_rng():
        torch.manual_seed(37)
        first = sampler._do(problem, 128)
        second = sampler._do(problem, 128)
        torch.manual_seed(37)
        replay = sampler._do(problem, 128)
    assert first != second
    assert first == replay
    points = np.array([[row[name] for name in space.ordered_params] for row in first])
    # GA real coordinates are normalized; integers retain their discrete range.
    assert np.all((points[:, 0] >= 0) & (points[:, 0] <= 1))
    assert np.all((points[:, 1] >= 1) & (points[:, 1] <= 7))
    np.testing.assert_array_equal(points[:, 1], np.round(points[:, 1]))


def test_ga_evaluates_128_new_offspring_per_generation():
    class Acquisition:
        n_obj = 3

        def __init__(self):
            self.batch_sizes = []

        def __call__(self, x):
            self.batch_sizes.append(len(x))
            x = x.squeeze(-2)
            return torch.stack((x[:, 0], x[:, 1], -x.sum(-1)), dim=-1)

    space = MixedDesignSpace([RealP(0, 0, 1), RealP(1, 0, 1)])
    acquisition = Acquisition()
    front = optimize_real_hedged_acquisition(
        acquisition, space, torch.empty((0, 2), dtype=torch.double),
        config=GASearchConfig(pop_size=96, max_generations=3),
    )
    assert acquisition.batch_sizes == [96, 128, 128]
    assert front.shape[1] == 2 and len(front) > 0
    assert torch.isfinite(front).all()
