from dataclasses import replace
import pytest
import torch
from tidyhebo.layered import leader_config, ScalarizationConfig, LayeredSearchConfig
from tidyhebo.layered.scalarization import Scalarizer


def test_retained_settings_are_explicit_and_immutable():
    cfg = leader_config()
    assert cfg.acquisition.q == 4 and cfg.acquisition.mc_samples == 512
    assert cfg.model.input_warp and cfg.model.output_transform == "yeo_johnson"
    s = cfg.search
    assert s.points.algorithm == "initial_only"
    assert s.point_init.directions == s.batch_init.directions == 6
    assert s.point_init.restarts == s.batch_init.restarts == 3
    assert s.batches.population == 48 and s.batches.generations == 20
    assert s.batch_init.pool_generations == 5 and s.batch_init.use_point_pool
    assert s.batches.memetic == "end" and s.matching_distance == "kernel"
    assert s.scalarization.rho == 0.01
    from tidyhebo.config import TidyHEBOConfig, GASearchConfig

    assert isinstance(TidyHEBOConfig().search, GASearchConfig)


def test_augmented_chebyshev_has_frozen_sobol_scale_and_correct_gradient():
    raw = torch.tensor([[0.0, 2.0, 4.0], [2.0, 6.0, 10.0]], dtype=torch.double)
    scalar = Scalarizer(raw, ScalarizationConfig())
    score = torch.tensor([[1.3, 3.0, 9.0]], dtype=torch.double, requires_grad=True)
    ray = torch.tensor([0.0, 0.4, 0.6], dtype=torch.double)
    w = 1 / ray.clamp_min(0.02)
    terms = (1 - (score - raw.amin(0)) / (raw.amax(0) - raw.amin(0)) + 0.05) * w
    expected = terms.amax(-1) + 0.01 * terms.sum(-1)
    torch.testing.assert_close(scalar(score, ray), expected)
    assert torch.autograd.grad(scalar(score, ray).sum(), score)[0].isfinite().all()


@pytest.mark.parametrize("override", [dict(evaluation_chunk=0), dict(matching_distance="posterior")])
def test_unsupported_settings_fail(override):
    with pytest.raises(ValueError):
        LayeredSearchConfig(**override)
