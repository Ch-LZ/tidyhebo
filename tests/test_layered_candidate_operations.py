"""Candidate math regression: same values, gradients, callback order and RNG."""

import math

import numpy as np
import pytest
import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from tidyhebo.layered.candidate_operations import (
    canonical_gp, evaluate_in_chunks, feasible_batches, unique_batches,
    rank_scores, refinement_targets,
)
from tidyhebo.qmace.ga_search import canonicalize


def _rng():
    return np.random.get_state(), torch.random.get_rng_state().clone()


def _assert_rng_equal(first, second):
    a, b = first[0], second[0]
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2:] == b[2:]
    assert torch.equal(first[1], second[1])


def _original_canonical(x):
    if x.shape[-2] < 2:
        return x
    for j in range(x.shape[-1] - 1, -1, -1):
        order = torch.argsort(x[..., j], dim=-1, stable=True)
        x = torch.gather(x, -2, order.unsqueeze(-1).expand_as(x))
    return x


@pytest.mark.parametrize('q', [1, 2, 4])
def test_canonical_gp_values_and_gradients(q):
    x = torch.linspace(.05, .95, 3*q*2, dtype=torch.double).reshape(3,q,2)
    x = x.flip(-2).requires_grad_(True)
    old = _original_canonical(x)
    new = canonical_gp(x)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(old.square().sum(), x, retain_graph=True)[0],
        torch.autograd.grad(new.square().sum(), x)[0], rtol=0, atol=0,
    )


@pytest.mark.parametrize('q', [1, 2, 4])
def test_feasibility_matches_previous_constraints_and_does_not_draw_rng(q):
    torch.manual_seed(314)
    x = torch.rand(7, q, 3, dtype=torch.double)
    x[0, 0, 0] = -0.1
    if q > 1:
        x[1, 1] = x[1, 0]
    observed = torch.rand(5, 3, dtype=torch.double)
    x[2, 0] = observed[0]
    tol = torch.full((3,), 1e-5, dtype=torch.double)
    before = _rng()
    detached = x.detach()
    valid = torch.isfinite(detached).all((-1, -2)) & (detached >= 0).all((-1,-2)) & (detached <= 1).all((-1,-2))
    if len(observed):
        novel = ((detached.unsqueeze(-2)-observed).abs()>tol).any(-1).all((-1,-2))
        valid &= novel
    pairwise = ((detached.unsqueeze(-2)-detached.unsqueeze(-3)).abs()>tol).any(-1)
    diagonal = torch.eye(detached.shape[-2],dtype=torch.bool,device=detached.device)
    expected = valid & (pairwise | diagonal).all((-1,-2))
    result = feasible_batches(x, observed, tol)
    assert torch.equal(result,expected)
    _assert_rng_equal(before,_rng())


class _IdentitySpace:
    def from_GP_to_GA(self, x):
        return x


def test_uniqueness_first_appearance_and_no_rng():
    x = torch.tensor([
        [[.1,.2],[.3,.4]], [[.3,.4],[.1,.2]], [[.5,.6],[.8,.9]],
        [[.1,.2],[.3,.4]]
    ], dtype=torch.double)
    space = _IdentitySpace()
    before = _rng()
    ga = space.from_GP_to_GA(x.detach()).cpu().numpy()
    seen, keep = set(), []
    for i,b in enumerate(ga):
        key = canonicalize(b).tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    assert torch.equal(unique_batches(x,space),x[keep])
    assert torch.equal(unique_batches(x[:0],space),x[:0])
    _assert_rng_equal(before,_rng())


def test_rank_maximized_objectives():
    scores = np.array([[2.,1.,0.], [0.,3.,1.], [2.,3.,1.], [1.,1.,0.]])
    before=_rng()
    actual=rank_scores(scores)
    expected=NonDominatedSorting().do(-scores)
    assert len(actual)==len(expected)
    for a,b in zip(actual,expected):
        np.testing.assert_array_equal(a,b)
    _assert_rng_equal(before,_rng())


def test_acquisition_evaluation_preserves_chunks_gradients_and_rng():
    def make_acq(log):
        def score(x):
            log.append(tuple(x.shape))
            # Deliberately draw randomness to detect any change in call order.
            noise = torch.rand(len(x), 1, dtype=x.dtype)
            scale = np.random.random()
            return (x.square().sum((-1,-2)).unsqueeze(-1) + noise) * scale
        return score

    x=torch.rand(9,3,2,dtype=torch.double,requires_grad=True)
    reference=torch.zeros(1,dtype=torch.double)
    np.random.seed(314);torch.manual_seed(314)
    old_log=[]
    old=torch.cat([make_acq(old_log)(_original_canonical(part.to(reference)))
                   for part in x.split(4)],0)
    old_grad=torch.autograd.grad(old.sum(),x,retain_graph=True)[0]
    expected_rng=_rng()
    np.random.seed(314);torch.manual_seed(314)
    new_log=[]
    new=evaluate_in_chunks(make_acq(new_log),x,reference,4)
    new_grad=torch.autograd.grad(new.sum(),x)[0]
    assert old_log==new_log
    assert torch.equal(old,new)
    assert torch.equal(old_grad,new_grad)
    _assert_rng_equal(expected_rng,_rng())


def test_last_generation_refinement_target_order():
    values=torch.tensor([[1.,3.,2.],[2.,3.,1.],[4.,1.,0.],[3.,2.,4.]],dtype=torch.double)
    directions=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[.5,.3,.2]])
    def scalar(values,ray):
        return -(values*torch.as_tensor(ray).to(values)).sum(-1)
    n=max(1,math.ceil(len(values)*.6))
    selected,rays=[],[]
    for ray in directions:
        index=int(scalar(values,ray).argmin())
        if index not in selected:
            selected.append(index)
            rays.append(ray)
    expected_indices,expected_rays=selected[:n],rays[:n]
    before=_rng()
    actual_indices,actual_rays=refinement_targets(values,directions,scalar,.6)
    assert expected_indices==actual_indices
    np.testing.assert_array_equal(np.asarray(expected_rays),np.asarray(actual_rays))
    _assert_rng_equal(before,_rng())
