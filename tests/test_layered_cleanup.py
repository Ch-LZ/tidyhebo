"""Final cleanup: removed adapters must preserve Sobol/cleanup and RNG."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tidyhebo.layered.candidate_operations import (
    canonical_gp,
    feasible_batches,
    snap_to_design_space,
    unique_batches,
)
from tidyhebo.layered.search import LayeredSearch


class _RoundedSpace:
    """Simple quantizer with the same GP→GA→GP conversion contract."""

    def from_GP_to_GA(self, x):
        return torch.round(x * 100) / 100

    def from_GA_to_GP(self, x):
        return x


def _rng():
    return np.random.get_state(), torch.random.get_rng_state().clone()


def _same_rng(actual, expected):
    a, b = actual[0], expected[0]
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2:] == b[2:]
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("q", [1, 2, 4])
def test_sobol_same_samples_and_rng_after_removing_round_adapter(q):
    state = SimpleNamespace(q=q, d=2, ref=torch.zeros(1, dtype=torch.double), space=_RoundedSpace())
    np.random.seed(439)
    torch.manual_seed(439)
    raw = torch.quasirandom.SobolEngine(state.d * state.q, scramble=True).draw(7, dtype=state.ref.dtype)
    raw = raw.to(state.ref).reshape(7, state.q, state.d)
    expected = state.space.from_GA_to_GP(state.space.from_GP_to_GA(raw))
    expected_rng = _rng()

    np.random.seed(439)
    torch.manual_seed(439)
    actual = LayeredSearch.sobol(state, 7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _same_rng(_rng(), expected_rng)


@pytest.mark.parametrize("q", [1, 2, 4])
def test_clean_matches_previous_round_and_unique_adapters(q):
    torch.manual_seed(439)
    space = _RoundedSpace()
    observed = torch.tensor([[0.1, 0.1]], dtype=torch.double)
    tolerances = torch.tensor([1e-4, 1e-4], dtype=torch.double)
    x = torch.rand(8, q, 2, dtype=torch.double, requires_grad=True)
    x = torch.cat((x, x[[0]], x[[1]]), dim=0)
    state = SimpleNamespace(
        space=space, observed=observed, tolerances=tolerances,
        feasible=lambda batches: feasible_batches(batches, observed, tolerances),
    )

    # Old LayeredSearch.clean: round -> canonicalize -> feasible -> unique.
    rounded = space.from_GA_to_GP(space.from_GP_to_GA(x.detach()))
    ordered = canonical_gp(rounded)
    expected = unique_batches(ordered[feasible_batches(ordered, observed, tolerances)], space)
    expected_rng = _rng()

    actual = LayeredSearch.clean(state, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad
    _same_rng(_rng(), expected_rng)
    torch.testing.assert_close(
        snap_to_design_space(x, space), space.from_GA_to_GP(space.from_GP_to_GA(x)),
        rtol=0, atol=0,
    )
