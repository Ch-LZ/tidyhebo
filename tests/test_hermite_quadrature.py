import math
import warnings

import pytest
import torch
from botorch.models import SingleTaskGP

from tidyhebo.acquisition import QuadratureMACE
from tidyhebo.config import AcquisitionConfig
from tidyhebo.yeo_johnson_warp import YeoJohnsonGPytorch


@pytest.mark.parametrize("size", [512, 1024])
def test_large_hermite_quadrature_is_finite_and_preserves_gaussian_moments(size):
    x = torch.linspace(0, 1, 9, dtype=torch.double).unsqueeze(-1)
    y = torch.sin(5 * x)
    model = SingleTaskGP(x, y)
    warp = YeoJohnsonGPytorch()
    warp.lam = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        acquisition = QuadratureMACE(
            model=model, best_f=y.max(), train_x=x, outcome_warp=warp,
            config=AcquisitionConfig(evaluation="quadrature", quadrature_size=size, mc_samples=16),
        )

    nodes = acquisition.hermite_nodes
    weights = acquisition.hermite_weights
    assert nodes.shape == weights.shape == (size,)
    assert torch.isfinite(nodes).all() and torch.isfinite(weights).all()
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(), nodes.new_tensor(math.sqrt(math.pi)))
    # Under exp(-x**2) / sqrt(pi), E[x] = 0 and E[x**2] = 1/2.
    torch.testing.assert_close((weights * nodes).sum(), nodes.new_tensor(0.0))
    torch.testing.assert_close(
        (weights * nodes.square()).sum() / weights.sum(), nodes.new_tensor(0.5),
    )
