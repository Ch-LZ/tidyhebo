"""
To test correctness of Yeo-Johnson implementation
"""
import torch
import pytest
from pytest import approx
from sklearn.preprocessing import PowerTransformer

from tidyhebo.yeo_johnson_warp import (
    yeo_johnson_transform, inv_yeo_johnson_transform,
    YeoJohnsonGPytorch,
    )


def _helper(lam, sign):
    lam = lam
    # deterministic
    y = sign * torch.tensor([0.0, 1e-15, 1.0, 2.0, 0.5, 0.6123]).double()
    y_tr = yeo_johnson_transform(y.clone(), lam=lam)
    y_recovered = inv_yeo_johnson_transform(y_tr, lam=lam)
    assert y == approx(y_recovered)

    # random
    y = sign * torch.rand(100, 1).double()
    y_tr = yeo_johnson_transform(y.clone(), lam=lam)
    y_recovered = inv_yeo_johnson_transform(y_tr, lam=lam)
    assert y == approx(y_recovered)


def test_positive_y_genral():
    _helper(lam=2.0, sign=1.0)
    _helper(lam=-2.0, sign=1.0)


def test_positive_y_special():
    _helper(lam=0.0, sign=1.0)


def test_negative_y_general():
    _helper(lam=1.0, sign=-1.0)
    _helper(lam=-1.0, sign=-1.0)


def test_negative_y_special():
    _helper(lam=2, sign=-1.0)


def test_likelihood():    
    y = torch.rand(100, 1).double()
    # lambda 1
    reference_power = PowerTransformer(method="yeo-johnson")
    reference_power.fit(y)
    lam_reference = reference_power.lambdas_
    # assert False, lam_reference
    
    # lambda 2
    my_power = YeoJohnsonGPytorch(prior=None)
    my_power._optimize(y.ravel())
    lam_got = my_power.lam

    assert lam_reference[0] == approx(lam_got.item(), abs=1e-3)
