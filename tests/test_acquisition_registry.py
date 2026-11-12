"""Registered acquisitions preserve native scores and their shared contract."""

import pytest
import torch
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.acquisition import qLogExpectedImprovement, qProbabilityOfImprovement, qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler

from tidyhebo.acquisition import MACE, AnalyticMACE, QuadratureMACE, EI, LogEI
from tidyhebo.base_acquisition import BaseAcquisition
from tidyhebo.config import AcquisitionConfig
from tidyhebo.yeo_johnson_warp import YeoJohnsonGPytorch
from tests.test_qmace import gp_model


@pytest.mark.parametrize(
    "target,cls,q", [("MACE", MACE, 1), ("qMACE", MACE, 2), ("EI", EI, 1), ("LogEI", LogEI, 1)]
)
def test_registered_scores_shapes_and_gradients(target, cls, q):
    model = gp_model()
    config = AcquisitionConfig(
        _target_=target,
        q=q,
        mc_samples=32,
        qmc_seed=11,
        evaluation="qmc" if target in ("MACE", "qMACE") else "analytic",
    )
    best = model.train_targets.max()
    acq = BaseAcquisition.from_config(config, model=model, best_f=best, train_x=model.train_inputs[0])
    assert isinstance(acq, cls) and acq.config is config and acq.model is model
    assert acq.q == q and acq.n_objectives == acq.n_obj == (3 if target in ("MACE", "qMACE") else 1)
    X = torch.rand(2, 3, q, 2, dtype=torch.double, requires_grad=True)
    if target in ("MACE", "qMACE"):

        def sampler():
            return SobolQMCNormalSampler(torch.Size([32]), seed=11)

        expected = torch.stack(
            (
                qLogExpectedImprovement(model, best, sampler=sampler())(X),
                qProbabilityOfImprovement(model, best, sampler=sampler())(X),
                qUpperConfidenceBound(model, config.beta, sampler=sampler())(X),
            ),
            -1,
        )
    else:
        native = ExpectedImprovement if target == "EI" else LogExpectedImprovement
        expected = native(model, best_f=best)(X).unsqueeze(-1)
    values = acq(X)
    assert values.shape == (2, 3, acq.n_obj)
    torch.testing.assert_close(values, expected)
    assert torch.isfinite(torch.autograd.grad(values.sum(), X)[0]).all()


def test_scalar_acquisition_rejects_nonlinear_original_scale():
    with pytest.raises(ValueError, match="Gaussian outcomes"):
        BaseAcquisition.from_config(
            AcquisitionConfig(_target_="LogEI"),
            model=gp_model(),
            best_f=1.0,
            train_x=None,
            outcome_warp=YeoJohnsonGPytorch(),
        )


@pytest.mark.parametrize("target", ["EI", "LogEI"])
def test_scalar_acquisition_flows_through_optimizer_and_discrete_search(target):
    import numpy as np
    import pandas as pd
    from tidyhebo import tidyHEBO, RealP
    from tidyhebo.config import TidyHEBOConfig, ModelConfig, DiscreteSearchConfig

    x = pd.DataFrame({"x0": [0.0, 0.3, 0.6, 1.0]})
    candidates = pd.DataFrame({"x0": [0.15, 0.45, 0.85]})
    optimizer = tidyHEBO(
        [RealP(0, 0, 1)],
        train_x=x,
        train_y=np.sin(x.to_numpy()),
        config=TidyHEBOConfig(
            model=ModelConfig(output_transform="standardize", input_warp=False),
            acquisition=AcquisitionConfig(_target_=target),
            search=DiscreteSearchConfig(candidates=candidates),
        ),
    )
    selected = optimizer.suggest()
    assert len(selected) == 1 and selected.iloc[0, 0] in candidates.x0.values


@pytest.mark.parametrize(
    "q,evaluation,warped,expected",
    [
        (1, "analytic", False, AnalyticMACE),
        (1, "qmc", False, MACE),
        (1, "qmc", True, MACE),
        (1, "quadrature", True, QuadratureMACE),
        (4, "qmc", True, MACE),
    ],
)
def test_mace_factory_selects_explicit_evaluator(q, evaluation, warped, expected):
    model = gp_model()
    warp = YeoJohnsonGPytorch() if warped else None
    cfg = AcquisitionConfig(
        _target_="MACE" if q == 1 else "qMACE", q=q, evaluation=evaluation, quadrature_size=32, mc_samples=16
    )
    acq = BaseAcquisition.from_config(
        cfg, model=model, best_f=0.0, train_x=model.train_inputs[0], outcome_warp=warp
    )
    assert type(acq) is expected and acq.config is cfg
    if expected is not MACE:
        assert not hasattr(acq, "qLogEI")


def test_explicit_quadrature_rejects_missing_warp_and_joint_batches():
    model = gp_model()
    with pytest.raises(ValueError, match="active Yeo-Johnson"):
        BaseAcquisition.from_config(AcquisitionConfig(evaluation="quadrature"), model=model, best_f=0.0)
    with pytest.raises(ValueError, match="q == 1"):
        QuadratureMACE(
            model, AcquisitionConfig(_target_="qMACE", q=2), best_f=0.0, outcome_warp=YeoJohnsonGPytorch()
        )


def test_quadrature_triplet_evaluates_posterior_once(monkeypatch):
    model = gp_model()
    acq = QuadratureMACE(
        model,
        AcquisitionConfig(evaluation="quadrature", quadrature_size=32),
        best_f=0.0,
        outcome_warp=YeoJohnsonGPytorch(),
    )
    calls = []
    posterior = model.posterior

    def counted(*args, **kwargs):
        calls.append(1)
        return posterior(*args, **kwargs)

    monkeypatch.setattr(model, "posterior", counted)
    assert acq(torch.rand(4, 1, 2, dtype=torch.double)).shape == (4, 3)
    assert len(calls) == 1


def test_analytic_triplet_matches_native_values_and_coordinate_gradients():
    from botorch.acquisition import ProbabilityOfImprovement, UpperConfidenceBound

    model = gp_model()
    best = model.train_targets.max()
    cfg = AcquisitionConfig(evaluation="analytic")
    acq = BaseAcquisition.from_config(cfg, model=model, best_f=best)
    x = torch.rand(2, 3, 1, 2, dtype=torch.double, requires_grad=True)
    expected = torch.stack(
        (
            LogExpectedImprovement(model, best)(x),
            ProbabilityOfImprovement(model, best)(x),
            UpperConfidenceBound(model, cfg.beta)(x),
        ),
        -1,
    )
    actual = acq(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x, retain_graph=True)[0], torch.autograd.grad(expected.sum(), x)[0]
    )
    with pytest.raises(ValueError, match="Gaussian outcomes"):
        BaseAcquisition.from_config(cfg, model=model, best_f=best, outcome_warp=YeoJohnsonGPytorch())


def test_evaluation_defaults_are_fixed_and_auto_is_rejected():
    assert AcquisitionConfig().evaluation == "qmc"
    assert AcquisitionConfig(_target_="LogEI").evaluation == "analytic"
    with pytest.raises(ValueError, match="evaluation"):
        AcquisitionConfig(evaluation="auto")
