"""
To test correctness of Yeo-Johnson implementation
"""
import torch
import pytest
from pytest import approx
from sklearn.preprocessing import PowerTransformer

from tidyhebo.yeo_johnson_warp import YeoJohnsonGPytorch


class YJNormal(YeoJohnsonGPytorch):

    def _neg_log_like(self, y0):
        """
        Loss for YJ(y) ~ N(mu, sigma) likelihood
        DON'T use while training with GP simultaneously.
        For testing purposes. 
        """

        if y0.ndim != 2:
            y0 = y0.unsqueeze(-1)

        x_trans = self(y0)
        n_samples = x_trans.shape[0]
        x_trans_var = x_trans.var(axis=0)

        log_var = torch.log(x_trans_var)
        loglike = -n_samples / 2 * log_var
        # note: loss with one lambda and no shift
        loglike += self.log_jacobian(y0)
        # loglike += (self.lam - 1) * (torch.sign(y0) * torch.log1p(torch.abs(y0))).sum()

        return -loglike


    def _optimize(self, y0):
        """
        For YJ(y) ~ N(mu, sigma) likelihood. Mainly for testing purposes.
        """
        from torch.optim import Adam
        optimizer = Adam(self.parameters(), lr=1e-2)
        for i in range(1000):
            optimizer.zero_grad()

            loss = self._neg_log_like(y0)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.lam.clamp_(0.0, 2.0)

        self.eval()


def test_likelihood():    
    y = torch.rand(100, 1).double()
    # lambda 1
    reference_power = PowerTransformer(method="yeo-johnson")
    reference_power.fit(y)
    lam_reference = reference_power.lambdas_
    # assert False, lam_reference
    
    # lambda 2
    my_power = YJNormal()
    my_power._optimize(y.ravel())
    lam_got = my_power.lam

    assert lam_reference[0] == approx(lam_got.item(), abs=1e-3)
