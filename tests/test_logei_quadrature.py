"""Log-domain scores must retain small tails and their useful gradients."""
import pytest
import torch
from scipy.special import ndtri_exp

from tidyhebo.acquisition import MACE, QuadratureMACE
from tidyhebo.config import AcquisitionConfig
from tidyhebo.yeo_johnson_warp import YeoJohnsonGPytorch
from tests.test_qmace import gp_model


def acquisition(evaluation="quadrature", lam=1.0):
    model = gp_model()
    warp = YeoJohnsonGPytorch()
    warp.lam = lam
    cls = MACE if evaluation == "qmc" else QuadratureMACE
    return cls(model, AcquisitionConfig(evaluation=evaluation,
                quadrature_size=256, mc_samples=32, qmc_seed=7),
                best_f=0.0, train_x=model.train_inputs[0], outcome_warp=warp)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse_log_cdf_matches_scipy_and_has_finite_gradients(dtype):
    logp = torch.tensor([-0.001, -1., -19.99, -20.01, -100., -1000., -10000.],
                        dtype=dtype, requires_grad=True)
    actual = QuadratureMACE._normal_quantile_from_log_prob(logp)
    expected = torch.from_numpy(ndtri_exp(logp.detach().double().numpy())).to(dtype)
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual.sum(), logp)[0]
    assert torch.isfinite(grad).all() and (grad > 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_identity_quadrature_matches_analytic_logei_including_deep_tails(dtype):
    acq = acquisition().to(dtype=dtype)
    mean = torch.tensor([-40., -12., -8., -5., -1., 0., 2.], dtype=dtype, requires_grad=True)
    sigma = torch.ones_like(mean, requires_grad=True)
    actual = acq._logei(mean, sigma)
    # Native analytic Gaussian LogEI is an independent stable reference.
    from botorch.acquisition.analytic import _log_ei_helper
    expected = _log_ei_helper(mean / sigma) + sigma.log()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-4)
    grads = torch.autograd.grad(actual.sum(), (mean, sigma))
    assert all(torch.isfinite(g).all() and (g > 0).all() for g in grads)
    assert actual[0] < -800  # EI itself would underflow, even in float64.


def test_mace_preserves_native_log_scores():
    acq = acquisition(evaluation="qmc", lam=1.3)
    X = torch.tensor([[[.2, .3]], [[.6, .8]]], dtype=torch.double, requires_grad=True)
    values = acq(X)
    torch.testing.assert_close(values[..., 0], acq.qLogEI(X))
    assert torch.isfinite(torch.autograd.grad(values.sum(), X)[0]).all()


@pytest.mark.parametrize("lam", [0., .5, 1., 1.5, 2.])
def test_warped_quadrature_log_scores_and_gradients(lam):
    acq = acquisition(lam=lam)
    X = torch.tensor([[[.2, .3]], [[.6, .8]]], dtype=torch.double, requires_grad=True)
    values = acq(X)
    assert torch.isfinite(values).all()
    assert torch.isfinite(torch.autograd.grad(values.sum(), X)[0]).all()
