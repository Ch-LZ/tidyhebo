"""Behavioral checks for joint q-acquisition, unordered variation and public IO."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.acquisition import qLogExpectedImprovement, qProbabilityOfImprovement, qUpperConfidenceBound
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.kernels import RBFKernel
from pymoo.core.individual import Individual
from pymoo.core.problem import Problem

from tidyhebo import tidyHEBO
from tidyhebo.config import TidyHEBOConfig, ModelConfig, AcquisitionConfig, GASearchConfig, DiscreteSearchConfig
from tidyhebo.archive import ParetoAccumulator
from tidyhebo.design_space import MixedDesignSpace
from tidyhebo.mixed_params import BoolP, CategoricalP, IntegerP, RealP
from tidyhebo.acquisition import MACE
from tidyhebo.qmace import optimize_qmace
from tidyhebo.evolution_optimizer import BatchLayout
from tidyhebo.qmace.ga_search import BatchProblem, BatchSampler, canonicalize, distinct_points
from tidyhebo.qmace.ga_components import BatchMatcher, SetDuplicates, SetMating
from tidyhebo.yeo_johnson_warp import YeoJohnsonGPytorch


def real_space():
    return MixedDesignSpace([RealP(0, -2, 3), RealP(1, 0, 1)])


def gp_model():
    x = torch.tensor([[0., 0.], [.25, .8], [.6, .2], [1., 1.]], dtype=torch.double)
    y = torch.tensor([[0.], [1.], [.4], [-.2]], dtype=torch.double)
    warp = Warp(d=2, indices=[0, 1]).double()
    with torch.no_grad():
        warp.concentration0.copy_(torch.tensor([1.2, .8], dtype=torch.double))
        warp.concentration1.copy_(torch.tensor([.8, 1.3], dtype=torch.double))
    kernel = RBFKernel(ard_num_dims=2).double()
    kernel.lengthscale = torch.tensor([.2, .7], dtype=torch.double)
    model = SingleTaskGP(x, y, input_transform=warp, covar_module=kernel).double()
    model.eval()
    return model


@pytest.mark.parametrize("q", [2, 4, 8])
@pytest.mark.parametrize("warped", [False, True])
def test_joint_values_match_native_log_botorch_triplet(q, warped):
    model = gp_model()
    objective = None
    warp = None
    if warped:
        warp = YeoJohnsonGPytorch()
        warp.lam = .7
        objective = GenericMCObjective(lambda samples, X: warp.untransform(samples).squeeze(-1))
    acquisition = MACE(model, AcquisitionConfig(_target_="qMACE", q=q, mc_samples=64, qmc_seed=17),
                        best_f=.5, outcome_warp=warp)
    X = torch.rand(3, q, 2, dtype=torch.double, generator=torch.Generator().manual_seed(13))

    def sampler():
        return SobolQMCNormalSampler(torch.Size([64]), seed=17)

    # Preserve the native logarithmic improvement score in the triplet.
    reference = torch.stack((
        qLogExpectedImprovement(model, .5, sampler=sampler(), objective=objective)(X),
        qProbabilityOfImprovement(model, .5, sampler=sampler(), objective=objective)(X),
        qUpperConfidenceBound(model, 4., sampler=sampler(), objective=objective)(X),
    ), dim=-1)
    torch.testing.assert_close(acquisition(X), reference)
    torch.testing.assert_close(acquisition(X), acquisition(X))
    assert acquisition(X).shape == (3, 3)
    assert acquisition(X.reshape(1, 3, q, 2)).shape == (1, 3, 3)
    X.requires_grad_()
    assert torch.isfinite(torch.autograd.grad(acquisition(X).sum(), X)[0]).all()


@pytest.mark.parametrize("q", [1, 3])
def test_unordered_mixed_batches_roundtrip_and_duplicates(q):
    space = MixedDesignSpace([RealP(0, 0, 1), IntegerP(1, 1, 4), CategoricalP(2, ["a", "b", "c"])])
    layout = BatchLayout(space, q)
    A = np.array([[.8, 2, 2], [.2, 4, 3], [.4, 1, 1]])[:q]
    np.testing.assert_array_equal(layout.from_records(layout.to_records(A)), A)
    dedup = SetDuplicates(layout)
    assert dedup.is_equal(Individual(X=layout.to_records(A)), Individual(X=layout.to_records(A[::-1])))
    B = A.copy(); B[0, 0] = .7
    assert not dedup.is_equal(Individual(X=layout.to_records(A)), Individual(X=layout.to_records(B)))
    assert layout.from_records([]).shape == (0, q, 3)
    np.testing.assert_array_equal(layout.from_records([layout.to_records(A)]), A[None])
    records = layout.to_records(A[None])
    assert records == [layout.to_records(A)]
    np.testing.assert_array_equal(layout.from_records(np.asarray(records, dtype=object)), A[None])
    np.testing.assert_array_equal(layout.from_records(tuple(records)), A[None])
    assert layout.to_records(np.empty((0, q, 3))) == []
    with pytest.raises(ValueError, match="GA coordinates"):
        layout.to_records(np.zeros((q, 4)))
    for slot in range(q):
        for j, p in enumerate(space.params):
            assert type(layout.vars[layout.names[slot][j]]) is type(p.pymoo_var)


@pytest.mark.parametrize("distance", ["euclidean", "kernel"])
def test_matching_and_mating_ignore_parent_permutations(distance):
    space, model = real_space(), gp_model()
    matcher = BatchMatcher(space, model, distance)
    A = np.array([[.1, .2], [.8, .9], [.45, .55]])
    B = np.array([[.82, .88], [.43, .58], [.12, .18]])
    expected = matcher.align(A, B)
    actual = matcher.align(A[[2, 0, 1]], B[[1, 2, 0]])
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    D = matcher.distance_matrix(A, A)
    np.testing.assert_allclose(D, D.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(D), 0, atol=1e-10)

    layout = BatchLayout(space, 3)
    mating = SetMating(layout, matcher)
    problem = Problem(vars=layout.vars)
    outputs = []
    for left, right in [(A, B), (A[[2, 0, 1]], B[[1, 2, 0]])]:
        parents = [[Individual(X=layout.to_records(left)), Individual(X=layout.to_records(right))]]
        np.random.seed(19)
        outputs.append(mating._do(problem, parents, 2, parents=True, random_state=np.random.default_rng(19)))
    for a, b in zip(*outputs):
        np.testing.assert_array_equal(layout.from_records(a.X), layout.from_records(b.X))


class Tradeoffs:
    n_obj = 3

    def __init__(self, q):
        self.q, self.model, self.sizes = q, gp_model(), []

    def __call__(self, X):
        self.sizes.append(len(X))
        return torch.stack((X[..., 0].mean(-1), 1 - X[..., 0].mean(-1), X[..., 1].max(-1).values), -1)


def test_batch_feasibility_requires_novel_and_distinct_points():
    space = MixedDesignSpace([RealP(0, 0, 1), IntegerP(1, 0, 3)])
    layout = BatchLayout(space, 2)
    batches = np.array([[[.1, 2], [.8, 3]], [[.1, 2], [.1, 2]], [[.2, 1], [.8, 3]]])
    observed = torch.tensor([[.2, 1]], dtype=torch.double)
    problem = BatchProblem(Tradeoffs(2), space, layout, observed)
    output = {}
    problem._evaluate([layout.to_records(batch) for batch in batches], output)
    assert (output["G"] <= 0).tolist() == [True, False, False]
    assert output["F"].shape == (3, 3)


@pytest.mark.parametrize("distance", ["euclidean", "kernel"])
def test_real_search_archive_matches_full_feasible_history(monkeypatch, distance):
    import tidyhebo.qmace.ga_search as search
    histories, archives = [], []

    class CheckedArchive(ParetoAccumulator):
        def __init__(self):
            super().__init__(); archives.append(self)

        def add(self, X, F, feasible=None):
            mask = feasible.reshape(-1)
            histories.append((X[mask].copy(), F[mask].copy()))
            super().add(X, F, feasible)
            all_x = np.concatenate([x for x, _ in histories])
            all_f = np.concatenate([f for _, f in histories])
            keep = is_non_dominated(-torch.from_numpy(all_f), deduplicate=False).numpy()
            expected = {tuple(row) for row in np.concatenate((all_x, all_f), axis=1)[keep]}
            actual = {tuple(row) for row in np.concatenate((self.X, self.F), axis=1)}
            assert actual == expected

    monkeypatch.setattr(search, "ParetoAccumulator", CheckedArchive)
    acquisition, space = Tradeoffs(2), real_space()
    observed = torch.tensor([[.2, .3]], dtype=torch.double)
    batches = optimize_qmace(acquisition, space, observed,
        config=GASearchConfig(pop_size=96, max_generations=3, matching_distance=distance))
    assert acquisition.sizes == [96, 128, 128]
    assert batches.ndim == 3 and batches.shape[1:] == (2, 2) and len(batches)
    assert distinct_points(batches, space).all()
    assert space.are_values_new(batches.flatten(0, 1), observed).all()
    assert not archives[0].has_data
    for batch in batches.numpy():
        np.testing.assert_array_equal(batch, canonicalize(batch))


def test_scrambled_batch_initialization_replays_with_seed():
    space = real_space(); layout = BatchLayout(space, 4)
    sampler = BatchSampler(space, layout); problem = Problem(vars=layout.vars)
    torch.manual_seed(37); first = sampler._do(problem, 96)
    second = sampler._do(problem, 96)
    torch.manual_seed(37); replay = sampler._do(problem, 96)
    assert first == replay and first != second


def test_initialization_returns_remaining_points_without_fitting():
    optimizer = tidyHEBO([RealP(0, 0, 1)], n_init=4,
        config=TidyHEBOConfig(acquisition=AcquisitionConfig(_target_="qMACE", q=2)))
    first = optimizer.suggest()
    optimizer.observe(first, np.zeros((2, 1)))
    optimizer.config = replace(optimizer.config, acquisition=replace(optimizer.config.acquisition, q=8))
    last = optimizer.suggest()
    assert len(first) == len(last) == 2
    assert len(pd.concat((first, last)).drop_duplicates()) == 4
    assert optimizer.model is None


@pytest.mark.parametrize("q,distance", [(2, "euclidean"), (4, "kernel"), (8, "euclidean")])
def test_public_joint_suggest_observe_and_all_batches(q, distance):
    torch.manual_seed(41); np.random.seed(41)
    optimizer = tidyHEBO([RealP(0, -2, 3), BoolP(1), IntegerP(2, 1, 4),
                          CategoricalP(3, ["a", "b", "c"])], n_init=4,
        config=TidyHEBOConfig(
            model=ModelConfig(output_transform="standardize" if q == 4 else "yeo_johnson"),
            acquisition=AcquisitionConfig(_target_="qMACE", q=q, mc_samples=32, qmc_seed=12),
            search=GASearchConfig(pop_size=96, max_generations=2, matching_distance=distance),
            return_all=True,
        ))
    initial = optimizer.suggest()
    optimizer.observe(initial, -(initial.iloc[:, 0].to_numpy() ** 2)[:, None])
    optimizer.config = replace(optimizer.config, return_all=False)
    points = optimizer.suggest()
    assert len(points) == q and points.dtypes.iloc[1] == bool
    assert pd.api.types.is_integer_dtype(points.dtypes.iloc[2])
    assert set(points.iloc[:, 3]) <= {"a", "b", "c"}
    assert len(points.drop_duplicates()) == q
    ga = optimizer.design_space.from_DS_to_GA(points)
    assert optimizer.design_space.are_values_new(ga, optimizer.train_x).all()
    optimizer.observe(points, -(points.iloc[:, 0].to_numpy() ** 2)[:, None])
    assert len(optimizer.train_x) == 4 + q
    optimizer.config = replace(optimizer.config, return_all=True)
    front = optimizer.suggest()
    assert isinstance(front.index, pd.MultiIndex)
    assert front.index.names == ["batch", "point"]
    assert (front.groupby(level="batch").size() == q).all()
    assert optimizer.design_space.are_values_new(optimizer.design_space.from_DS_to_GA(front), optimizer.train_x).all()


def test_fallback_respects_discrete_exhaustion_and_tensor_format():
    space = MixedDesignSpace([BoolP(0)])
    empty = torch.empty((0, 1), dtype=torch.double)
    batch = space.sample_unobserved(empty, q=2, max_draws=32).unsqueeze(0)
    assert set(batch.flatten().tolist()) == {1., 2.}
    with pytest.raises(RuntimeError, match="novel fallback"):
        space.sample_unobserved(torch.tensor([[1.]], dtype=torch.double), q=2, max_draws=32)


def test_invalid_batch_configs_fail_before_gp_training():
    for q in [0, -1, True, 2.5]:
        with pytest.raises(ValueError, match="integer"):
            AcquisitionConfig(_target_="qMACE", q=q)
    with pytest.raises(NotImplementedError, match="q=1"):
        tidyHEBO([RealP(0, 0, 1)], n_init=4, config=TidyHEBOConfig(
            acquisition=AcquisitionConfig(_target_="qMACE", q=2),
            search=DiscreteSearchConfig(candidates=pd.DataFrame({"x0": [.1, .2]})),
        ))
    with pytest.raises(ValueError, match="matching_distance"):
        BatchMatcher(real_space(), gp_model(), distance="invalid")


@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("return_all", [False, True])
def test_suggest_preserves_front_elements_and_output_format(monkeypatch, q, return_all):
    from types import SimpleNamespace
    from tidyhebo.candidate_search import CandidateSearch

    parameters = [RealP(0, -2, 3), BoolP(1)]
    observed = pd.DataFrame({"x0": [0.], "x1": [False]})
    optimizer = tidyHEBO(parameters, train_x=observed,
        train_y=[[0.]],
        config=TidyHEBOConfig(return_all=return_all,
            acquisition=AcquisitionConfig(_target_="MACE" if q == 1 else "qMACE", q=q)))
    front = torch.tensor([[[.1, 1.], [.2, 2.]], [[.7, 2.], [.8, 1.]]], dtype=torch.double)[:, :q]
    search = SimpleNamespace(search=lambda *args: front[:, 0] if q == 1 else front)
    monkeypatch.setattr(CandidateSearch, "from_config", lambda *args, **kwargs: search)
    monkeypatch.setattr(optimizer, "_train_and_get_acqf", lambda: (None, None, None))
    result = optimizer.suggest()
    assert isinstance(result, pd.DataFrame)
    assert result.dtypes.iloc[1] == bool
    if return_all and q > 1:
        assert result.index.names == ["batch", "point"]
    else:
        assert not isinstance(result.index, pd.MultiIndex)
    actual = optimizer.design_space.from_DS_to_GA(result)
    if return_all:
        assert result.shape == (2 * q, 2)
        torch.testing.assert_close(actual.reshape_as(front), front)
    else:
        assert result.shape == (q, 2)
        assert any(torch.allclose(actual, batch) for batch in front)
