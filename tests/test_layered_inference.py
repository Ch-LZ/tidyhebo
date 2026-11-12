from dataclasses import replace
import pytest
import torch
from tests.test_layered_search import source_and_space
from tidyhebo.layered.inference import inference_copy, resolve_device
from tidyhebo.layered.search import canonical_gp


@pytest.mark.parametrize("q", [1, 2, 4, 8])
@pytest.mark.parametrize("seed", [19, None])
def test_frozen_copy_preserves_native_values_gradients_dtype_and_rng(q, seed):
    cfg, source, space, observed = source_and_space(q)
    if seed is None:
        from tidyhebo.acquisition import MACE

        source = MACE(source.model, replace(source.config, qmc_seed=None), source.best_f)
    before = {k: v.clone() for k, v in source.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    cloned = inference_copy(source, torch.device("cpu"))
    assert torch.equal(rng, torch.get_rng_state())
    x = torch.rand(4, q, 2, dtype=torch.double, requires_grad=True)
    expected = source(canonical_gp(x))
    actual = cloned(canonical_gp(x))
    torch.testing.assert_close(expected, actual, rtol=1e-9, atol=1e-9)
    g1 = torch.autograd.grad(expected.sum(), x, retain_graph=True)[0]
    g2 = torch.autograd.grad(actual.sum(), x)[0]
    torch.testing.assert_close(g1, g2, rtol=1e-7, atol=1e-8)
    assert all(not p.requires_grad and p.dtype == torch.double for p in cloned.model.parameters())
    assert all(p.requires_grad and p.device.type == "cpu" for p in source.model.parameters())
    for key, value in source.model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_explicit_cuda_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA inference requested"):
        resolve_device("cuda")
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(ValueError):
        resolve_device("mps")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("q", [1, 2, 4, 8])
def test_cuda_values_gradients_and_complete_search(q):
    cfg, source, space, x = source_and_space(q)
    cuda = inference_copy(source, torch.device("cuda"))
    z = torch.rand(5, q, 2, dtype=torch.double, requires_grad=True)
    a = source(canonical_gp(z))
    b = cuda(canonical_gp(z.to("cuda"))).cpu()
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(a.sum(), z)[0], torch.autograd.grad(b.sum(), z)[0], rtol=1e-5, atol=1e-6
    )
    from tidyhebo.candidate_search import CandidateSearch

    search = CandidateSearch.from_config(replace(cfg.search, inference_device="cuda"), design_space=space)
    result = search.search(source, x)
    assert result.device.type == "cpu" and result.dtype == torch.double
    assert result.shape[-2:] == (q, 2) if q > 1 else result.shape[-1] == 2
    assert search.ref.device.type == "cuda"
