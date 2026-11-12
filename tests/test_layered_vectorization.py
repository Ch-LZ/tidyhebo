import numpy as np
import torch
from tests.test_layered_search import source_and_space
from tidyhebo.layered.search import LayeredSearch


def test_grouped_restarts_match_separate_anchor_and_direction_optimizations():
    cfg, source, space, x = source_and_space(2)
    stage = LayeredSearch(space, cfg.search)
    stage._prepare_level(source, x, cfg.search.batches)
    raw = stage.clean(stage.sobol(16))
    rays = np.array([stage.directions[0], stage.directions[-1], stage.directions[1], stage.directions[0]])
    objectives = [0, -1, -1, 2]
    independent = torch.cat(
        [
            stage._optimize_targets(raw, np.asarray(ray)[None], [obj], 2, 4)
            for ray, obj in zip(rays, objectives)
        ]
    )
    grouped = stage._optimize_targets(raw, rays, objectives, 2, 4)
    torch.testing.assert_close(grouped, independent, atol=2e-8, rtol=1e-7)
    assert not grouped.requires_grad


def test_best_feasible_per_restart_never_worsens_directional_loss():
    cfg, source, space, x = source_and_space(4)
    stage = LayeredSearch(space, cfg.search)
    stage._prepare_level(source, x, cfg.search.batches)
    starts = stage.clean(stage.sobol(8))
    rays = torch.tensor(stage.directions, dtype=torch.double)
    rays = rays[torch.arange(len(starts)) % len(rays)]
    with torch.no_grad():
        before = stage.scalar(stage.scores(starts), rays)
    refined = stage._refine(starts, rays, 4)
    with torch.no_grad():
        after = stage.scalar(stage.scores(refined), rays)
    assert (after <= before + 1e-9).all() and stage.feasible(refined).all()
