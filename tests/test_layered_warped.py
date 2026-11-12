from dataclasses import replace
import numpy as np
import pytest
import torch
from tidyhebo import tidyHEBO, RealP
from tidyhebo.design_space import MixedDesignSpace
from tidyhebo.layered.inference import inference_copy
from tidyhebo.layered.search import canonical_gp
from tests.test_layered_search import small_config


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
        ),
    ],
)
@pytest.mark.parametrize("q", [1, 4])
def test_real_yj_input_warp_gp_copy_and_suggest(device, q):
    torch.manual_seed(31)
    np.random.seed(31)
    params = [RealP(0, 0.1, 2), RealP(1, -1, 3)]
    space = MixedDesignSpace(params)
    x = torch.tensor([[0.1, 0.1], [0.3, 0.2], [0.5, 0.8], [0.7, 0.3], [0.9, 0.9]], dtype=torch.double)
    y = torch.tensor([[0.1], [0.25], [0.9], [0.3], [1.1]], dtype=torch.double)
    cfg = small_config(q)
    cfg = replace(
        cfg, model=replace(cfg.model, gate=False), search=replace(cfg.search, inference_device=device)
    )
    optimizer = tidyHEBO(
        params, train_x=space.from_GP_to_DS(x, return_df=True), train_y=y.numpy(), config=cfg
    )
    fitted, source, state = optimizer._train_and_get_acqf()
    assert fitted.use_power_outcome_transform
    weights = {k: v.detach().clone() for k, v in fitted.model.named_parameters()}
    clone = inference_copy(source, torch.device(device))
    z = torch.rand(3, q, 2, dtype=torch.double, requires_grad=True)
    a = source(canonical_gp(z))
    b = clone(canonical_gp(z.to(device))).cpu()
    torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(
        torch.autograd.grad(a.sum(), z)[0], torch.autograd.grad(b.sum(), z)[0], rtol=1e-5, atol=1e-6
    )
    # Use a fresh inference cache; public suggest still fits in float64 on CPU.
    batch = optimizer.suggest()
    assert len(batch) == q and np.isfinite(batch.to_numpy()).all()
    assert optimizer.train_x.device.type == "cpu" and optimizer.train_y.dtype == torch.double
    assert all(p.device.type == "cpu" and p.requires_grad for p in fitted.model.parameters())
    for k, v in fitted.model.named_parameters():
        torch.testing.assert_close(v, weights[k])
