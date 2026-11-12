"""Fit the YJ normality likelihood against an independent bounded reference."""
import numpy as np
import pytest
import torch
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import yeojohnson_llf

from tidyhebo.yeo_johnson_warp import YeoJohnsonGPytorch


class YJNormal(YeoJohnsonGPytorch):
    def _neg_log_like(self, y):
        """Normality NLL with fitted mean/variance, without constants or prior."""
        transformed = self(y)
        return y.numel() / 2 * transformed.var(correction=0).log() - self.log_jacobian(y)

    def _optimize(self, y):
        """Fit the constrained raw parameter with autograd and SciPy L-BFGS-B."""
        def closure(raw):
            with torch.no_grad():
                self.raw_lam.copy_(self.raw_lam.new_tensor(raw[0]))
            loss = self._neg_log_like(y)
            gradient, = torch.autograd.grad(loss, self.raw_lam)
            return loss.item(), np.array([gradient.item()])

        result = minimize(
            closure, np.array([self.raw_lam.item()]), jac=True, method="L-BFGS-B",
            options={"maxiter": 300, "gtol": 1e-9, "ftol": 1e-12},
        )
        with torch.no_grad():
            self.raw_lam.copy_(self.raw_lam.new_tensor(result.x[0]))
        self.eval()
        return result


@pytest.mark.parametrize('case', ['interior', 'lower_bound', 'upper_bound'])
def test_likelihood(case):
    if case == 'interior':
        y = torch.rand(100, generator=torch.Generator().manual_seed(0), dtype=torch.double)
    else:
        y = torch.linspace(-4, 4, 100, dtype=torch.double).exp()
        if case == 'upper_bound':
            y = -y
    data = y.numpy()
    reference = minimize_scalar(
        lambda lam: -yeojohnson_llf(lam, data), bounds=(0., 2.), method='bounded',
        options={'xatol': 1e-12},
    )
    assert reference.success
    # A bounded scalar search approaches endpoints but does not include them.
    expected_lambda = min((0., reference.x, 2.), key=lambda lam: -yeojohnson_llf(lam, data))
    expected_loss = -yeojohnson_llf(expected_lambda, data)
    if case == 'interior':
        assert 0 < expected_lambda < 2
    else:
        assert expected_lambda == (0 if case == 'lower_bound' else 2)

    fitted = YJNormal()
    result = fitted._optimize(y)
    assert result.success, result.message
    assert fitted.lam.item() == pytest.approx(expected_lambda, abs=1e-5)
    assert fitted._neg_log_like(y).item() == pytest.approx(expected_loss, abs=1e-7)
