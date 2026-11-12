"""L1 regression checks: original q=1 factory and Pareto-pool selection."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tidyhebo.base_acquisition import BaseAcquisition
from tidyhebo.layered.config import LevelConfig
from tidyhebo.layered.point_search_l1 import make_point_acquisition, select_point_pool


@dataclass(frozen=True)
class _SourceConfig:
    _target_: str = "qMACE"
    q: int = 4
    evaluation: str = "qmc"


def _rng_snapshot():
    return np.random.get_state(), torch.random.get_rng_state().clone()


def _check_rng(expected):
    numpy_expected, torch_expected = expected
    numpy_actual, torch_actual = _rng_snapshot()
    assert numpy_actual[0] == numpy_expected[0]
    np.testing.assert_array_equal(numpy_actual[1], numpy_expected[1])
    assert numpy_actual[2:] == numpy_expected[2:]
    assert torch.equal(torch_actual, torch_expected)


@pytest.mark.parametrize("warped", [False, True])
def test_point_acquisition_matches_original_factory_calls_and_rng(monkeypatch, warped):
    source = SimpleNamespace(
        config=_SourceConfig(), model=object(), best_f=object(),
        outcome_warp=object() if warped else None,
    )
    calls = []

    def fake_factory(config, **kwargs):
        # The real factory may allocate QMC samplers; preserve its call order.
        calls.append((config, kwargs))
        return (np.random.randint(0, 10000), torch.rand(1).item())

    monkeypatch.setattr(BaseAcquisition, "from_config", staticmethod(fake_factory))
    np.random.seed(410)
    torch.manual_seed(410)
    before = _rng_snapshot()
    evaluation = "quadrature" if source.outcome_warp is not None else "analytic"
    point_config = replace(source.config, _target_="MACE", q=1, evaluation=evaluation)
    old_result = BaseAcquisition.from_config(
        point_config, model=source.model, best_f=source.best_f, outcome_warp=source.outcome_warp
    )
    old_calls = list(calls)
    after = _rng_snapshot()
    np.random.set_state(before[0])
    torch.random.set_rng_state(before[1])
    calls.clear()
    result = make_point_acquisition(source)
    assert result == old_result
    assert calls == old_calls
    _check_rng(after)


@pytest.mark.parametrize("deeper,pool_size", [(False, 2), (True, 4), (True, 12)])
def test_pareto_pool_matches_original_logic_and_preserves_rng(deeper, pool_size):
    level = replace(LevelConfig(), deeper_fronts=deeper, pool_size=pool_size)
    candidates = torch.arange(18, dtype=torch.double).reshape(9, 1, 2)
    fronts = [np.array([3, 2]), np.array([6, 0, 7]), np.array([1, 8, 4, 5])]
    np.random.seed(411)
    torch.manual_seed(411)
    before = _rng_snapshot()

    ids = []
    for front in fronts:
        ids.extend(front.tolist())
        if not level.deeper_fronts or len(ids) >= level.pool_size:
            break
    old_result = candidates[ids[: level.pool_size]]
    after = _rng_snapshot()

    result = select_point_pool(candidates, fronts, level)
    assert torch.equal(result, old_result)
    _check_rng(after)
    assert torch.equal(before[1], after[1])
    np.testing.assert_array_equal(before[0][1], after[0][1])
