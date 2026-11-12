from dataclasses import replace
import pytest
import torch
import tidyhebo.base_model as module
from tidyhebo.base_model import ProfiledModel, SimpleModel
from tidyhebo.config import ModelConfig
from test_centered_model import data, parameters, loss


@pytest.mark.parametrize("transform", ["yeo_johnson", "standardize"])
@pytest.mark.parametrize("input_warp", [False, True])
def test_profile_matches_joint_at_gls_mean_in_value_and_gradients(
    transform, input_warp
):
    x, y = data()
    config = ModelConfig(input_warp=input_warp, gate=False, output_transform=transform)
    profiled = ProfiledModel(x, y, replace(config, _target_="profiled"))
    profiled.build_gp_mll()
    actual = loss(profiled)
    assert not profiled.model.mean_module.raw_constant.requires_grad
    target = (
        profiled.model.outcome_warp_yj(y).squeeze(-1)
        if transform == "yeo_johnson"
        else profiled.model.train_targets
    )
    # Independent dense GLS reference, intentionally not the production whitening formula.
    covariance = profiled.model.likelihood(profiled.model(x)).covariance_matrix
    solved = torch.linalg.solve(
        covariance, torch.stack((target, torch.ones_like(target)), -1)
    )
    mean = solved[:, 0].sum() / solved[:, 1].sum()
    torch.testing.assert_close(profiled.model.mean_module.constant, mean)
    joint = SimpleModel(x, y, config)
    joint.build_gp_mll()
    joint.model.load_state_dict(profiled.model.state_dict())
    reference = loss(joint)
    torch.testing.assert_close(actual, reference, atol=1e-9, rtol=1e-9)
    keys = [n for n, p in profiled.model.named_parameters() if p.requires_grad]
    expected = torch.autograd.grad(
        reference, [dict(joint.model.named_parameters())[n] for n in keys]
    )
    observed = torch.autograd.grad(
        actual, [dict(profiled.model.named_parameters())[n] for n in keys]
    )
    for a, b in zip(observed, expected):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("transform", ["yeo_johnson", "standardize"])
def test_real_profiled_fit_and_posterior_match_joint_at_returned_parameters(
    monkeypatch, transform
):
    x, y = data()
    config = ModelConfig(
        input_warp=False, gate=False, output_transform=transform, max_attempts=1
    )
    profiled = ProfiledModel(x, y, config)
    state = profiled.train_model()
    assert all(torch.isfinite(p).all() for p in state.values())
    monkeypatch.setattr(module, "fit_gpytorch_mll", lambda mll, **kwargs: mll)
    joint = SimpleModel(x, y, config)
    joint.train_model(state)
    query = x.new_tensor([[0.2], [0.7]])
    with torch.no_grad():
        a, b = profiled.model.posterior(query), joint.model.posterior(query)
    torch.testing.assert_close(a.mean, b.mean)
    torch.testing.assert_close(
        a.distribution.covariance_matrix, b.distribution.covariance_matrix
    )


@pytest.mark.parametrize("transform", ["yeo_johnson", "standardize"])
def test_final_profile_is_recomputed_after_solver_restores_parameters(
    monkeypatch, transform
):
    x, y = data()
    config = ModelConfig(input_warp=False, gate=False, output_transform=transform)
    profiled = ProfiledModel(x, y, config)

    def solve(mll, **kwargs):
        loss(profiled)
        with torch.no_grad():
            if transform == "yeo_johnson":
                profiled.model.outcome_warp_yj.lam = 0.7
            profiled.model.mean_module.constant.fill_(999.0)
        return mll

    monkeypatch.setattr(module, "fit_gpytorch_mll", solve)
    state = profiled.train_model()
    # Compare with an independent dense GLS solve at the FINAL parameters.
    # Merely checking "finite and not 999" would accept an incorrect zero mean.
    with torch.no_grad():
        profiled.model.train()  # Obtain the GP prior, not its conditioned posterior.
        covariance = profiled.model.likelihood(profiled.model(x)).covariance_matrix
        target = (
            profiled.model.outcome_warp_yj(y).squeeze(-1)
            if transform == "yeo_johnson"
            else profiled.model.train_targets
        )
        weights = torch.linalg.solve(covariance, torch.ones_like(target))
        expected = weights.dot(target) / weights.sum()
    torch.testing.assert_close(state["mean_module.raw_constant"], expected)
    torch.testing.assert_close(profiled.model.mean_module.constant, expected)
