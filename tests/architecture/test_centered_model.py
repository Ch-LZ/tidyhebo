from dataclasses import replace

import pytest
import torch

import tidyhebo.base_model as module
from tidyhebo.base_model import CenteredModel, SimpleModel, SurrogateModel
from tidyhebo.config import ModelConfig


def data():
    x = torch.linspace(0.05, 0.95, 6, dtype=torch.double)[:, None]
    y = torch.tensor([-2.0, -0.4, 0.3, 1.7, 2.0, 4.0], dtype=torch.double)[:, None]
    return x, y


def parameters(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def loss(wrapper):
    return -wrapper.mll(wrapper.model(wrapper.train_x), wrapper.model.train_targets)


@pytest.mark.parametrize("transform", ["yeo_johnson", "standardize"])
@pytest.mark.parametrize("input_warp", [False, True])
def test_loss_and_gradients_match_the_change_of_coordinates(
    monkeypatch, transform, input_warp
):
    x, y = data()
    config = ModelConfig(input_warp=input_warp, gate=False, output_transform=transform)
    joint = SimpleModel(x, y, config)
    joint.build_gp_mll()
    with torch.no_grad():
        joint.model.mean_module.constant.fill_(1.25)
        if transform == "yeo_johnson":
            joint.model.outcome_warp_yj.lam = 0.7
    state = parameters(joint.model)
    centered = CenteredModel(x, y, replace(config, _target_="centered"))

    def solve(mll, **kwargs):
        # Preserve physical m, but optimize beta during the solver call.
        torch.testing.assert_close(
            centered.model.mean_module.constant + centered._target_mean(),
            joint.model.mean_module.constant,
        )
        reference, actual = loss(joint), loss(centered)
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
        joint_params = dict(joint.model.named_parameters())
        centered_params = dict(centered.model.named_parameters())
        expected = dict(
            zip(
                joint_params,
                torch.autograd.grad(reference, tuple(joint_params.values())),
            )
        )
        observed = dict(
            zip(
                centered_params,
                torch.autograd.grad(actual, tuple(centered_params.values())),
            )
        )
        if transform == "yeo_johnson":
            key = "outcome_warp_yj.raw_lam"
            parameter = centered_params[key]
            (shift_grad,) = torch.autograd.grad(centered._target_mean(), parameter)
            correction = expected["mean_module.raw_constant"] * shift_grad
            assert correction.abs() > 1e-4  # A detached center must fail this test.
            expected[key] = expected[key] + correction
            # Independent finite difference at fixed beta.
            original = parameter.detach().clone()
            values = []
            for delta in (1e-5, -1e-5):
                with torch.no_grad():
                    parameter.copy_(original + delta)
                    values.append(loss(centered))
            with torch.no_grad():
                parameter.copy_(original)
            finite_difference = (values[0] - values[1]) / 2e-5
            torch.testing.assert_close(
                observed[key], finite_difference, atol=1e-7, rtol=1e-6
            )
        for name in expected:
            torch.testing.assert_close(
                observed[name], expected[name], atol=1e-9, rtol=1e-8
            )
        return mll

    monkeypatch.setattr(module, "fit_gpytorch_mll", solve)
    result = centered.train_model(state)
    assert not centered.mll.optimizing_beta
    torch.testing.assert_close(
        result["mean_module.raw_constant"], state["mean_module.raw_constant"]
    )


@pytest.mark.parametrize("init", ["preserve_mean", "zero_beta"])
@pytest.mark.parametrize("transform", ["yeo_johnson", "standardize"])
def test_warm_start_and_final_posterior_use_physical_mean(monkeypatch, init, transform):
    x, y = data()
    config = ModelConfig(input_warp=False, gate=False, output_transform=transform)
    previous = SimpleModel(x, y, config)
    previous.build_gp_mll()
    with torch.no_grad():
        previous.model.mean_module.constant.fill_(1.6)
    state = parameters(previous.model)
    # New observations must determine the center, not the preceding fit's buffers.
    new_x = torch.cat((x, x.new_tensor([[0.98]])))
    new_y = torch.cat((y * 2, y.new_tensor([[9.0]])))
    centered = SurrogateModel.from_config(
        replace(config, _target_="centered", centered_init=init),
        train_x=new_x,
        train_y=new_y,
    )

    def solve(mll, **kwargs):
        expected = (
            state["mean_module.raw_constant"] - centered._target_mean()
            if init == "preserve_mean"
            else new_y.new_tensor(0.0)
        )
        torch.testing.assert_close(centered.model.mean_module.constant, expected)
        with torch.no_grad():
            centered.model.mean_module.constant.fill_(0.35)
            if transform == "yeo_johnson":
                centered.model.outcome_warp_yj.lam = 1.3
        return mll

    monkeypatch.setattr(module, "fit_gpytorch_mll", solve)
    fitted_state = centered.train_model(state)
    physical_mean = 0.35 + centered._target_mean()
    torch.testing.assert_close(centered.model.mean_module.constant, physical_mean)
    torch.testing.assert_close(fitted_state["mean_module.raw_constant"], physical_mean)
    assert state["mean_module.raw_constant"].item() == 1.6
    # A plain GP at the same physical parameters must predict the same distribution.
    monkeypatch.setattr(module, "fit_gpytorch_mll", lambda mll, **kwargs: mll)
    reference = SimpleModel(new_x, new_y, config)
    reference.train_model(fitted_state)
    query = x.new_tensor([[0.15], [0.55], [0.85]])
    with torch.no_grad():
        actual, expected = (
            centered.model.posterior(query),
            reference.model.posterior(query),
        )
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected.distribution.covariance_matrix
    )
    # MLL evaluation after fitting must also use physical m, even in train mode.
    centered.mll.train()
    reference.mll.train()
    torch.testing.assert_close(loss(centered), loss(reference))


def test_failed_solver_restores_physical_mean(monkeypatch):
    x, y = data()
    model = CenteredModel(x, y, ModelConfig(input_warp=False, gate=False))

    def fail(mll, **kwargs):
        with torch.no_grad():
            model.model.mean_module.constant.fill_(0.45)
            model.model.outcome_warp_yj.lam = 0.8
        raise RuntimeError("solver failed")

    monkeypatch.setattr(module, "fit_gpytorch_mll", fail)
    with pytest.raises(RuntimeError, match="solver failed"):
        model.train_model()
    assert not model.mll.optimizing_beta
    torch.testing.assert_close(
        model.model.mean_module.constant, 0.45 + model._target_mean()
    )


def test_gate_refits_centered_model_without_yj(monkeypatch):
    x, y = data()
    model = CenteredModel(x, y, ModelConfig(input_warp=False))
    calls = []

    def solve(mll, **kwargs):
        calls.append(mll)
        assert mll.optimizing_beta
        if hasattr(model.model, "outcome_warp_yj"):
            model.model.outcome_warp_yj._scale = y.new_tensor(1e6)
        return mll

    monkeypatch.setattr(module, "fit_gpytorch_mll", solve)
    model.train_model()
    assert len(calls) == 2 and not hasattr(model.model, "outcome_warp_yj")
    assert all(not mll.optimizing_beta for mll in calls)
    assert model.config.output_transform == "yeo_johnson"


@pytest.mark.parametrize("transform", ["yeo_johnson", "standardize"])
def test_real_scipy_fit_produces_finite_predictions_and_parameter_state(transform):
    x, y = data()
    model = CenteredModel(
        x,
        y,
        ModelConfig(
            _target_="centered",
            input_warp=False,
            gate=False,
            output_transform=transform,
            max_attempts=1,
        ),
    )
    state = model.train_model()
    assert all(torch.isfinite(value).all() for value in state.values())
    assert not model.mll.optimizing_beta and not model.model.training
    with torch.no_grad():
        posterior = model.model.posterior(x.new_tensor([[0.2], [0.7]]))
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all() and (posterior.variance >= 0).all()
