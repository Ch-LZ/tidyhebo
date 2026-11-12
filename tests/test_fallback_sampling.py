"""Random fallback: novelty, bounded work, uniform proposals and no grid."""

import pytest
import torch

from tidyhebo.design_space import MixedDesignSpace
from tidyhebo.mixed_params import RealP, IntegerP


def test_batch_filters_history_and_selected_points(monkeypatch):
    space = MixedDesignSpace([IntegerP(0, 0, 3)])
    observed = torch.tensor([[1.]], dtype=torch.double)
    # GA proposals: observed 1, then 2, duplicate 2, then 3.
    gp = space.from_GA_to_GP(torch.tensor([[1.], [2.], [2.], [3.]], dtype=torch.double))
    monkeypatch.setattr(torch, 'rand', lambda *a, **kw: gp)
    result = space.sample_unobserved(observed, q=2, max_draws=4)
    torch.testing.assert_close(result, observed.new_tensor([[2.], [3.]]))


def test_refills_until_budget_is_exhausted(monkeypatch):
    space = MixedDesignSpace([RealP(0, 0, 1)])
    observed = torch.tensor([[.5]], dtype=torch.double)
    calls = []
    def draw(shape, **kwargs):
        calls.append(shape[0])
        return torch.full(shape, .5 if len(calls) == 1 else .9, **kwargs)
    monkeypatch.setattr(torch, 'rand', draw)
    result = space.sample_unobserved(observed, max_draws=300)
    assert calls == [256, 44]
    assert result.item() == .9
    calls.clear()
    with pytest.raises(RuntimeError, match='after 256 random draws'):
        space.sample_unobserved(observed, max_draws=256)
    assert calls == [256]


def test_failure_does_not_claim_proven_exhaustion():
    space = MixedDesignSpace([IntegerP(0, 0, 0)])
    with pytest.raises(RuntimeError, match='may be exhausted'):
        space.sample_unobserved(torch.ones((1, 1), dtype=torch.double), max_draws=17)


def test_high_dimension_fallback_does_not_build_cartesian_product(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('Cartesian product used')
    monkeypatch.setattr(torch, 'cartesian_prod', forbidden)
    space = MixedDesignSpace([RealP(i, 0, 1) for i in range(10)])
    old = torch.full((200, 10), .5, dtype=torch.double)
    result = space.sample_unobserved(old, q=16, max_draws=32)
    assert result.shape == (16, 10)
    assert space.are_values_new(result, old).all()
    for i in range(len(result)):
        assert space.are_values_new(result[i:i+1], result[:i]).all()


def test_seed_replay_and_uniform_conditional_distribution():
    space = MixedDesignSpace([RealP(0, 0, 1, ds_tolerance=.1)])
    old = torch.tensor([[.4]], dtype=torch.double)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        first = space.sample_unobserved(old)
        torch.manual_seed(17)
        assert torch.equal(first, space.sample_unobserved(old))
        # Allowed intervals: [0,.3) and (.5,1], with probability 3/8 on left.
        points = torch.cat([space.sample_unobserved(old, max_draws=32) for _ in range(2000)])
    assert space.are_values_new(points, old).all()
    assert (points < .3).double().mean().item() == pytest.approx(3 / 8, abs=.035)
