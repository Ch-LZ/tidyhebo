"""Search registry, coordinate boundaries, novelty and native scalar optimization."""

import pandas as pd
import pytest
import torch

from tidyhebo.candidate_search import CandidateSearch, DiscreteSearch, PymooSearch, BoTorchSearch
from tidyhebo.config import DiscreteSearchConfig, GASearchConfig, BoTorchSearchConfig
from tidyhebo.design_space import MixedDesignSpace
from tidyhebo.mixed_params import IntegerP, CategoricalP, RealP


@pytest.mark.parametrize('parameter,values', [
    (IntegerP(0, 0, 2, name='x'), [0, 2]),
    (CategoricalP(0, ['a', 'b', 'c'], name='x'), ['a', 'c']),
])
def test_discrete_coordinate_spaces_and_observation_filter(parameter, values):
    space = MixedDesignSpace([parameter])
    config = DiscreteSearchConfig(candidates=pd.DataFrame({'x': values}))
    search = CandidateSearch.from_config(config, design_space=space)
    assert isinstance(search, DiscreteSearch) and search.config is config
    ga = space.from_DS_to_GA(config.candidates)
    gp = space.from_GA_to_GP(ga)
    def acquisition(X):
        torch.testing.assert_close(X, gp[1:].unsqueeze(1))
        return X.new_ones((len(X), 3))
    torch.testing.assert_close(search.search(acquisition, ga[:1]), ga[1:])
    with pytest.raises(RuntimeError, match='unobserved'):
        search.search(acquisition, ga)


@pytest.mark.parametrize('q', [1, 2])
def test_pymoo_empty_result_uses_fallback(monkeypatch, q):
    import tidyhebo.candidate_search as module
    from types import SimpleNamespace
    space = MixedDesignSpace([RealP(0, 0, 1)])
    search = CandidateSearch.from_config(GASearchConfig(fallback_samples=16), design_space=space)
    assert isinstance(search, PymooSearch)
    monkeypatch.setattr(module, 'optimize_real_hedged_acquisition', lambda *a, **kw: torch.empty((0, 1)))
    monkeypatch.setattr(module, 'optimize_qmace', lambda *a, **kw: torch.empty((0, q, 1)))
    observed = torch.tensor([[.5]], dtype=torch.double)
    with pytest.warns(RuntimeWarning, match="GA returned no feasible candidates.*random fallback") as caught:
        result = search.search(SimpleNamespace(q=q), observed)
    assert len(caught) == 1
    assert result.numel() and space.are_values_new(result.reshape(-1, 1), observed).all()
    assert result.ndim == (2 if q == 1 else 3)


def test_botorch_search_optimizes_scalar_and_checks_compatibility():
    from types import SimpleNamespace
    space = MixedDesignSpace([RealP(0, -10, 20)])
    search = CandidateSearch.from_config(BoTorchSearchConfig(num_restarts=3, raw_samples=32), design_space=space)
    assert isinstance(search, BoTorchSearch)
    class Quadratic:
        q, n_obj = 1, 1
        def __call__(self, X):
            return -(X[..., 0, :] - .35).square().sum(-1, keepdim=True)
    result = search.search(Quadratic(), torch.empty((0, 1), dtype=torch.double))
    torch.testing.assert_close(result, result.new_tensor([[.35]]), atol=1e-5, rtol=0)
    with pytest.raises(ValueError, match='scalar'):
        search.search(SimpleNamespace(q=1, n_obj=3), result)
    with pytest.raises(ValueError, match='continuous'):
        CandidateSearch.from_config(BoTorchSearchConfig(),
            design_space=MixedDesignSpace([IntegerP(0, 0, 3)]))


@pytest.mark.parametrize('missing_population', [False, True])
def test_sequential_empty_population_reaches_fallback(monkeypatch, missing_population):
    from types import SimpleNamespace
    from pymoo.core.population import Population
    import tidyhebo.evolution_optimizer as evolution

    space = MixedDesignSpace([RealP(0, 0, 1)])
    search = PymooSearch(space, GASearchConfig())

    def minimize(*args, callback, **kwargs):
        import numpy as np
        # An infeasible generation creates an empty (0, d) archive. A later
        # empty population must not concatenate a rank-one empty X into it.
        callback(SimpleNamespace(pop=Population.new(
            'X', np.array([{space.ordered_params[0]: .5}]),
            'F', np.zeros((1, 3)), 'G', np.ones((1, 1)),
        )))
        callback(SimpleNamespace(pop=None if missing_population else Population.empty()))

    monkeypatch.setattr(evolution, 'minimize', minimize)
    observed = torch.tensor([[.5]], dtype=torch.double)
    with pytest.warns(RuntimeWarning, match="random fallback"):
        result = search.search(SimpleNamespace(q=1, n_obj=3), observed)
    assert result.ndim == 2 and len(result)
    assert space.are_values_new(result, observed).all()


@pytest.mark.parametrize('q', [1, 2])
def test_pymoo_success_does_not_warn_or_use_fallback(monkeypatch, q):
    from types import SimpleNamespace
    import warnings
    import tidyhebo.candidate_search as module
    space = MixedDesignSpace([RealP(0, 0, 1)])
    candidates = torch.tensor([[.3]], dtype=torch.double)
    if q > 1:
        candidates = torch.tensor([[[.3], [.7]]], dtype=torch.double)
    monkeypatch.setattr(module, 'optimize_real_hedged_acquisition', lambda *a, **kw: candidates)
    monkeypatch.setattr(module, 'optimize_qmace', lambda *a, **kw: candidates)
    def forbidden(*a, **kw):
        raise AssertionError('Fallback called after successful GA')
    monkeypatch.setattr(space, 'sample_unobserved', forbidden)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = PymooSearch(space, GASearchConfig()).search(
            SimpleNamespace(q=q), torch.empty((0, 1), dtype=torch.double))
    assert result is candidates and not caught
