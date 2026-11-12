from copy import deepcopy

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from tidyhebo.base_model import (
    SimpleModel,
    CenteredModel,
    ProfiledModel,
    _centered_gp_mll,
    _profiled_gp_mll,
)
from tidyhebo.config import ModelConfig
from tidyhebo.mll import BaseMLL, ExactMLL
from tidyhebo.yeo_johnson_warp import YJWarpedExactMLL
from test_centered_model import data


@pytest.mark.parametrize("transform", ["standardize", "yeo_johnson"])
@pytest.mark.parametrize(
    "model_class,calculation",
    [
        (SimpleModel, ExactMarginalLogLikelihood.forward),
        (CenteredModel, _centered_gp_mll),
        (ProfiledModel, _profiled_gp_mll),
    ],
)
def test_model_selects_calculation_independently_of_target_transform(
    transform,
    model_class,
    calculation,
):
    x, y = data()
    wrapper = model_class(
        x, y, ModelConfig(output_transform=transform, input_warp=False)
    )
    gp, mll = wrapper.build_gp_mll()
    expected_class = ExactMLL if transform == "standardize" else YJWarpedExactMLL
    assert type(mll) is expected_class
    assert isinstance(mll, BaseMLL) and mll.model is gp
    assert mll._gp_mll_impl is calculation
    assert not any("_gp_mll_impl" in name for name in mll.state_dict())


@pytest.mark.parametrize("transform", ["standardize", "yeo_johnson"])
def test_custom_calculation_receives_current_targets_and_jacobian_is_added_once(
    transform,
):
    x, y = data()
    seen = []

    def injected(mll, output, target, *params, **kwargs) -> torch.Tensor:
        seen.append((mll, target))
        assert kwargs.pop("di_marker") == "received"
        return (
            ExactMarginalLogLikelihood.forward(mll, output, target, *params, **kwargs)
            + 0.75
        )

    class InjectedModel(SimpleModel):
        def _create_mll(self) -> BaseMLL:
            return super()._create_mll(gp_mll=injected)

    wrapper = InjectedModel(
        x, y, ModelConfig(output_transform=transform, input_warp=False)
    )
    gp, mll = wrapper.build_gp_mll()
    if transform == "yeo_johnson":
        gp.outcome_warp_yj.lam = 0.7
    output = gp(x)
    target = (
        gp.outcome_warp_yj(y).squeeze(-1)
        if transform == "yeo_johnson"
        else gp.train_targets
    )
    expected = ExactMarginalLogLikelihood.forward(mll, output, target) + 0.75
    if transform == "yeo_johnson":
        expected = expected + gp.outcome_warp_yj.log_jacobian(y) / len(y)
    actual = mll(output, gp.train_targets, di_marker="received")
    assert len(seen) == 1 and seen[0][0] is mll
    torch.testing.assert_close(seen[0][1], target)
    torch.testing.assert_close(actual, expected)
    parameter_list = tuple(gp.parameters())
    observed = torch.autograd.grad(actual, parameter_list, retain_graph=True)
    reference = torch.autograd.grad(expected, parameter_list)
    for a, b in zip(observed, reference):
        torch.testing.assert_close(a, b)

    # A function stored in the MLL is not bound to the original MLL instance.
    copied = deepcopy(mll)
    copied(copied.model(x), copied.model.train_targets, di_marker="received")
    assert seen[-1][0] is copied and copied.model is not gp
