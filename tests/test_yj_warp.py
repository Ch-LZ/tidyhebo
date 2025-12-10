"""
To test correctness of Yeo-Johnson implementation
"""
import torch
import pytest
from pytest import approx

from tidyhebo.yeo_johnson_warp import (
    yeo_johnson_transform, inv_yeo_johnson_transform)


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
