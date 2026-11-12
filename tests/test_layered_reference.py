"""Check seeded replay and front invariants within one numerical environment.

An identical seed does not guarantee the same evolutionary trajectory across
operating systems, BLAS builds or thread settings. Tiny score differences can
change selection and thus the size of the final nondominated front. Historical
coordinate fixtures are archives, not portable correctness assertions.
"""

import random

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

from tidyhebo.candidate_search import CandidateSearch
from tidyhebo.qmace.ga_search import distinct_points
from tests.test_layered_search import source_and_space


def test_same_seed_replays_the_complete_layered_front():
    fronts = []
    for _ in range(2):
        random.seed(73)
        cfg, source, space, observed = source_and_space(4)  # Seeds NumPy and Torch.
        search = CandidateSearch.from_config(cfg.search, design_space=space)
        front = search.search(source, observed)
        fronts.append(front)

        assert front.ndim == 3 and front.shape[1:] == (cfg.acquisition.q, observed.shape[-1])
        assert 0 < len(front) <= cfg.search.batches.population
        assert front.dtype == torch.double and front.device.type == "cpu"
        assert torch.isfinite(front).all()
        assert ((front >= 0) & (front <= 1)).all()
        assert space.are_values_new(front.flatten(0, 1), observed).all()
        assert distinct_points(front, space).all()
        assert len(torch.unique(front.flatten(1), dim=0)) == len(front)

        # Check the actual AF used in this search, including its fixed QMC draws.
        with torch.no_grad():
            scores = search.scores(space.from_GA_to_GP(front))
        assert scores.shape == (len(front), 3) and torch.isfinite(scores).all()
        assert is_non_dominated(scores, deduplicate=False).all()

    # Same platform, settings and seeds must reproduce the full ordered result.
    torch.testing.assert_close(fronts[0], fronts[1], atol=1e-10, rtol=1e-10)
