import math
from unittest.mock import patch

import torch
from botorch.models import SingleTaskGP

from tidyhebo.acquisition import HedgedPareto
from tidyhebo.yeo_johnson_warp import YeoJohnsonGPytorch, YJWarpedExactMLL


def _training_data():
    train_x = torch.linspace(0.0, 1.0, 9, dtype=torch.double).unsqueeze(-1)
    train_y = (
        torch.sin(train_x * 5.0) + 0.3 * train_x.square()
    )
    return train_x, train_y


def test_yj_scale_is_nonpersistent_device_aware_buffer():
    train_x, train_y = _training_data()
    model = SingleTaskGP(train_x, train_y)
    warp = YeoJohnsonGPytorch()
    YJWarpedExactMLL(model.likelihood, model, warp, train_y)

    assert torch.allclose(warp._scale, train_y.std())
    assert "_scale" not in warp.state_dict()
    assert warp.to(dtype=torch.float32)._scale.dtype == torch.float32


def test_yj_mll_reuses_closure_model_output():
    train_x, train_y = _training_data()
    model = SingleTaskGP(train_x, train_y)
    warp = YeoJohnsonGPytorch()
    mll = YJWarpedExactMLL(model.likelihood, model, warp, train_y)
    model.train()
    mll.train()

    output = model(*model.train_inputs)
    with patch.object(model, "forward", wraps=model.forward) as model_forward:
        value = mll(output, model.train_targets)
        value.backward()

    model_forward.assert_not_called()
    assert warp.raw_lam.grad is not None
    assert torch.isfinite(warp.raw_lam.grad)


def test_fast_acquisition_matches_direct_posterior_sampling():
    torch.manual_seed(11)

    train_x, train_y = _training_data()
    warp = YeoJohnsonGPytorch()
    warp._scale = train_y.std()
    warp.lam = 1.3
    transformed_y = warp(train_y).detach()
    # Temporarily turn off input data and standardization validations
    import botorch
    with botorch.settings.validate_input_scaling(False):
        model = SingleTaskGP(train_x, transformed_y, outcome_transform=None)

    model.outcome_warp_yj = warp
    model.eval()

    best_f = float(train_y.max())
    beta = 4.0
    acquisition = HedgedPareto(
        model=model,
        best_f=best_f,
        beta=beta,
        train_x=train_x,
        use_logei=True,
        outcome_warp=warp,
        quadrature_size=64,
        mc_samples=32,
    )
    candidates = torch.linspace(0.05, 0.95, 7, dtype=torch.double).reshape(7, 1, 1)

    with torch.no_grad():
        fast_values = acquisition(candidates)
        posterior = model.posterior(candidates)
        latent_samples = posterior.rsample(torch.Size([2**17]))[..., 0, 0]
        outcome_samples = warp.untransform(latent_samples)
        outcome_mean = outcome_samples.mean(dim=0)
        reference = torch.stack(
            (
                (outcome_samples - best_f).clamp_min(0.0).mean(dim=0),
                (outcome_samples > best_f).double().mean(dim=0),
                outcome_mean
                + math.sqrt(beta * math.pi / 2)
                * (outcome_samples - outcome_mean).abs().mean(dim=0),
            ),
            dim=-1,
        )

    assert torch.allclose(fast_values[:, 0], reference[:, 0], atol=3e-3, rtol=3e-2)
    assert torch.allclose(fast_values[:, 1], reference[:, 1], atol=3e-3, rtol=3e-2)
    assert torch.allclose(fast_values[:, 2], reference[:, 2], atol=8e-3, rtol=1e-2)
