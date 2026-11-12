"""Independent value/gradient references for YJ and its removable singularities."""
import mpmath as mp
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.optim.closures import get_loss_closure
from gpytorch import settings

from tidyhebo.yeo_johnson_warp import (
    YeoJohnsonGPytorch, YJWarpedExactMLL,
    yeo_johnson_transform, inv_yeo_johnson_transform,
    _expm1_over_x, _log1p_over_x,
)


@pytest.fixture(params=[False, True], ids=['forward', 'inverse'])
def inverse(request):
    return request.param


def transform(inverse):
    return inv_yeo_johnson_transform if inverse else yeo_johnson_transform


def reference(y, lam, inverse):
    """70-digit, unexpanded definitions and analytic first derivatives.

    Return value, derivative by input, derivative by physical lambda.
    mpmath is already a GPyTorch dependency; no new production dependency.
    """
    with mp.workdps(70):
        y, lam = mp.mpf(float(y)), mp.mpf(float(lam))
        sign = 1 if y >= 0 else -1
        w = abs(y)
        a = lam if sign > 0 else 2 - lam
        if inverse:
            h = mp.log1p(a * w) / a if a else w
            value = sign * mp.expm1(h)
            dy = mp.exp(h) / (1 + a * w)
            dl = (mp.exp(h) * (a * w / (1 + a * w) - mp.log1p(a * w)) / a**2
                  if a else -mp.exp(w) * w**2 / 2)
        else:
            log = mp.log1p(w)
            value = sign * mp.expm1(a * log) / a if a else sign * log
            dy = mp.exp((a - 1) * log)
            dl = ((a * log * mp.exp(a * log) - mp.expm1(a * log)) / a**2
                  if a else log**2 / 2)
        return tuple(float(v) for v in (value, dy, dl))


def check_reference(y, lam, inverse):
    actual = transform(inverse)(y, lam)
    dy, dl = torch.autograd.grad(actual.sum(), (y, lam))
    # A lambda per element gives an independent reference for every gradient.
    expected = torch.tensor([reference(v.item(), a.item(), inverse)
                             for v, a in zip(y, lam)], dtype=y.dtype, device=y.device).T
    rtol, atol = ((3e-5, 3e-6) if y.dtype == torch.float32 else (2e-10, 2e-12))
    for value, target in zip((actual, dy, dl), expected):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, target, rtol=rtol, atol=atol)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('lam', [-3., -1e-12, 0., 1e-12, 1., 2-1e-12, 2., 2+1e-12, 3.])
def test_values_and_first_derivatives_against_high_precision(inverse, dtype, lam):
    values = [-.2, -1e-8, 0., 1e-8, .2] if inverse else [-1e3, -.25, 0., .25, 1e3]
    y = torch.tensor(values, dtype=dtype, requires_grad=True)
    parameter = torch.full_like(y, lam, requires_grad=True)
    check_reference(y, parameter, inverse)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('argument_sign', [-1, 1])
def test_values_and_gradients_across_series_cutoff(inverse, dtype, sign, argument_sign):
    # Construct arguments on both sides of the default cutoff. Compare each
    # point to an independent formula, rather than merely checking finiteness.
    eps = torch.finfo(dtype).eps
    cutoff = ((9 * eps / 8) ** (1 / 8) if inverse
              else (3991680 * eps) ** (1 / 10))
    y = torch.full((3,), sign * .75, dtype=dtype, requires_grad=True)
    argument = argument_sign * cutoff * torch.tensor([.99, 1., 1.01], dtype=dtype)
    magnitude = y.detach().abs() if inverse else y.detach().abs().log1p()
    a = argument / magnitude
    lam = (a if sign > 0 else 2 - a).requires_grad_()
    check_reference(y, lam, inverse)


@pytest.mark.parametrize('lam', [-1e-6, 0., 1e-6, 1., 2-1e-6, 2., 2+1e-6])
def test_gradcheck_by_input_and_lambda(inverse, lam):
    y = torch.tensor([-.2, 0., .4], dtype=torch.double, requires_grad=True)
    parameter = torch.tensor(lam, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(transform(inverse), (y, parameter),
                                    eps=1e-6, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('lam', [0., 2.])
def test_second_derivative_at_special_points(inverse, lam):
    # LBFGS uses first derivatives, but this catches a constant/linear-only
    # patch that hides a discontinuity in the derivative around the limit.
    y = torch.tensor([-.2, .4], dtype=torch.double, requires_grad=True)
    parameter = torch.tensor(lam, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradgradcheck(transform(inverse), (y, parameter),
                                       eps=1e-6, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('lam', [-2., -1e-8, 0., 1e-8, 1., 2-1e-8, 2., 2+1e-8, 3.])
def test_roundtrip_values_and_chain_rule(lam):
    y = torch.tensor([-2., -.2, 0., .2, 2.], dtype=torch.double, requires_grad=True)
    parameter = torch.tensor(lam, dtype=torch.double, requires_grad=True)
    recovered = inv_yeo_johnson_transform(yeo_johnson_transform(y, parameter), parameter)
    dy, dl = torch.autograd.grad(recovered.sum(), (y, parameter))
    torch.testing.assert_close(recovered, y, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(dy, torch.ones_like(y), rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(dl, torch.zeros_like(parameter), rtol=0, atol=1e-10)


@pytest.mark.parametrize('lam', [0., 1., 2.])
def test_log_jacobian_matches_input_derivative(lam):
    warp = YeoJohnsonGPytorch()
    warp.lam = lam
    warp._scale = torch.tensor(2.3, dtype=torch.double)
    y = torch.tensor([-.8, -.1, 0., .2, 1.1], dtype=torch.double, requires_grad=True)
    dy, = torch.autograd.grad(warp(y).sum(), y, create_graph=True)
    actual = dy.log().sum()
    # The existing MLL omits this lambda-independent scaling constant.
    expected = warp.log_jacobian(y) - y.numel() * warp._scale.log()
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    dl_actual, = torch.autograd.grad(actual, warp.raw_lam)
    dl_expected, = torch.autograd.grad(expected, warp.raw_lam)
    torch.testing.assert_close(dl_actual, dl_expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('lam', [1e-5, .01, 1., 1.99, 2-1e-5])
def test_full_mll_lambda_gradient_against_finite_difference(lam):
    x = torch.linspace(.05, .95, 6, dtype=torch.double).unsqueeze(-1)
    y = torch.tensor([[-.8], [.2], [1.1], [-.3], [.7], [.4]], dtype=torch.double)
    model = SingleTaskGP(x, y)
    warp = YeoJohnsonGPytorch()
    mll = YJWarpedExactMLL(model.likelihood, model, warp, y).double()
    mll.train()
    def set_lambda(value):
        warp.lam = warp.raw_lam.new_tensor(value)

    set_lambda(lam)
    with settings.fast_computations(covar_root_decomposition=False,
                                             log_prob=False, solves=False), \
            settings.max_cholesky_size(10000):
        closure = get_loss_closure(mll)
        loss = closure().sum()
        raw_gradient, = torch.autograd.grad(loss, warp.raw_lam)
        chain, = torch.autograd.grad(warp.lam, warp.raw_lam)
        physical_gradient = raw_gradient / chain
        # Central differences must remain inside the bounded physical interval.
        step = min(1e-4, lam / 4, (2 - lam) / 4)
        for step in (step, step / 10):
            try:
                with torch.no_grad():
                    set_lambda(lam + step)
                    plus = closure().sum()
                    set_lambda(lam - step)
                    minus = closure().sum()
            finally:
                set_lambda(lam)
            numerical = (plus - minus) / (2 * step)
            assert torch.isfinite(physical_gradient).all()
            torch.testing.assert_close(physical_gradient, numerical, rtol=2e-6, atol=2e-7)


def test_inverse_large_argument_does_not_overflow_unused_polynomial():
    y = torch.tensor([1e12], dtype=torch.float32, requires_grad=True)
    lam = torch.tensor(1., dtype=torch.float32, requires_grad=True)
    result = inv_yeo_johnson_transform(y, lam)
    dy, dl = torch.autograd.grad(result.sum(), (y, lam))
    assert torch.isfinite(dy).all() and torch.isfinite(dl).all()
    torch.testing.assert_close(result, y, rtol=3e-6, atol=0)


def test_inverse_out_of_domain_is_not_silently_clamped():
    assert torch.isnan(inv_yeo_johnson_transform(torch.tensor([1.]), -2.)).all()
    assert torch.isnan(inv_yeo_johnson_transform(torch.tensor([-1.]), 4.)).all()


def test_python_scalar_and_broadcast_lambda_preserve_values_and_gradients():
    y = torch.tensor([[-.2, 0., .4], [-.5, .1, .7]], dtype=torch.double, requires_grad=True)
    lam = torch.tensor([[0.], [2.]], dtype=torch.double, requires_grad=True)
    result = yeo_johnson_transform(y, lam)
    expected = torch.stack([yeo_johnson_transform(y[0], 0.), yeo_johnson_transform(y[1], 2.)])
    torch.testing.assert_close(result, expected)
    assert torch.autograd.gradcheck(yeo_johnson_transform, (y, lam))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('logarithmic', [False, True], ids=['exprel', 'log1prel'])
def test_helper_accuracy_near_balanced_series_cutoff(dtype, logarithmic):
    """Bound observed value/gradient error on both sides of the new cutoff."""
    eps = torch.finfo(dtype).eps
    cutoff = ((9 * eps / 8) ** (1 / 8) if logarithmic
              else (3991680 * eps) ** (1 / 10))
    x = (torch.linspace(-2, 2, 1025, dtype=dtype) * cutoff).requires_grad_()
    function = _log1p_over_x if logarithmic else _expm1_over_x
    actual = function(x, 1e-14)
    gradient, = torch.autograd.grad(actual.sum(), x)
    with mp.workdps(80):
        for argument, value, derivative in zip(x.detach().tolist(),
                                               actual.detach().tolist(),
                                               gradient.detach().tolist()):
            z = mp.mpf(argument)
            if not z:
                expected = mp.mpf(1)
                expected_gradient = mp.mpf('-.5' if logarithmic else '.5')
            elif logarithmic:
                expected = mp.log1p(z) / z
                expected_gradient = (z / (1 + z) - mp.log1p(z)) / z**2
            else:
                expected = mp.expm1(z) / z
                expected_gradient = (z * mp.exp(z) - mp.expm1(z)) / z**2
            assert abs(mp.mpf(value) - expected) < 4 * eps
            assert abs(mp.mpf(derivative) - expected_gradient) < 4 * eps / cutoff


@pytest.mark.parametrize('lam', [0., 1e-12, .01, 1., 1.99, 2-1e-12, 2.])
def test_inverse_value_only_path_against_high_precision(lam):
    """AF/GA calls need accurate values, including both lambda limits."""
    y = torch.tensor([-20., -1., -1e-12, 0., 1e-12, 1., 20.], dtype=torch.double)
    parameter = torch.tensor(lam, dtype=torch.double, requires_grad=True)
    with torch.no_grad():
        actual = inv_yeo_johnson_transform(y, parameter)
    # Also select the value-only path with autograd enabled but no active input.
    torch.testing.assert_close(actual, inv_yeo_johnson_transform(y, parameter.detach()))
    with mp.workdps(80):
        for value, output in zip(y.tolist(), actual.tolist()):
            z, a = mp.mpf(value), mp.mpf(lam)
            sign = 1 if z > 0 else -1
            a = a if sign > 0 else 2 - a
            w = abs(z)
            expected = sign * mp.expm1(mp.log1p(a * w) / a if a else w)
            assert abs(mp.mpf(output) - expected) <= mp.mpf('1e-14') * max(1, abs(expected))


def test_inverse_value_only_path_retains_domain_and_underflow_limit():
    with torch.no_grad():
        assert torch.isnan(inv_yeo_johnson_transform(torch.tensor([1.]), -2.)).all()
        assert torch.isnan(inv_yeo_johnson_transform(torch.tensor([-1.]), 4.)).all()
        y = torch.tensor([1e-100, .1], dtype=torch.double)
        for lam in (1e-300, 1e-320):
            torch.testing.assert_close(inv_yeo_johnson_transform(y, lam), y.expm1(),
                                       rtol=1e-14, atol=0)


def test_warp_bounded_lambda_starts_at_identity_and_rejects_invalid_values():
    warp = YeoJohnsonGPytorch()
    torch.testing.assert_close(warp.lam, torch.tensor(1., dtype=torch.double))
    y = torch.tensor([-10., -.1, 0., .1, 10.], dtype=torch.double)
    torch.testing.assert_close(warp(y), y)
    torch.testing.assert_close(warp.untransform(y), y)
    for value in [0., 1e-12, 1., 2-1e-12, 2.]:
        warp.lam = value
        torch.testing.assert_close(warp.lam, torch.tensor(value, dtype=torch.double),
                                   rtol=0, atol=3e-16)
        with torch.no_grad():
            assert torch.isfinite(warp.untransform(y)).all()
    warp.lam = 1.
    for invalid in [-.1, 2.1, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match=r'\[0, 2\]'):
            warp.lam = invalid
        torch.testing.assert_close(warp.lam, torch.tensor(1., dtype=torch.double))
    # Any finite raw parameter maps into the physical interval, independently
    # of the optimizer; saturation may round to an endpoint in floating point.
    for raw in [-1000., -10., 0., 10., 1000.]:
        with torch.no_grad():
            warp.raw_lam.fill_(raw)
        assert 0 <= warp.lam.item() <= 2
