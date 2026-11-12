"""Device placement for AF evaluation, independent of GP fitting.

The copy isolates prediction caches, frozen parameters and float64/device moves
from the optimizer's fitting model. Gradients remain enabled for candidate X.
No performance measurements or benchmark protocol belong in this module.
"""

import copy
import torch
from ..acquisition import MACE, SharedMACE
from ..base_acquisition import BaseAcquisition


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError, TypeError) as error:
        raise ValueError("inference_device must be cpu, auto, cuda or cuda:N") from error
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA inference is supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA inference requested but unavailable; install a CUDA Torch wheel and check the NVIDIA driver"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} does not exist")
    return device


def inference_copy(source, device):
    """Clone a fitted GP/AF without changing its weights, caches or RNG seeds.

    Parameters are frozen, but coordinate derivatives still flow through the
    posterior. A fresh prediction cache avoids copying non-leaf autograd data.
    """
    # Exclude prediction caches, which may contain non-leaf autograd tensors.
    cache = getattr(source.model, "prediction_strategy", None)
    model = copy.deepcopy(source.model, {id(cache): None} if cache is not None else {})
    model.to(device=device, dtype=torch.double)
    if getattr(model, "_original_train_inputs", None) is not None:
        model._original_train_inputs = model._original_train_inputs.to(device=device, dtype=torch.double)
    model.prediction_strategy = None
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    warp = getattr(model, "outcome_warp_yj", None) if source.outcome_warp is not None else None
    # Cloning must not consume the caller's RNG, including seed=None samplers.
    with torch.random.fork_rng(devices=[]):
        if isinstance(source, MACE):
            acq = SharedMACE(model, source.config, source.best_f.detach().to(device), outcome_warp=warp)
        else:
            acq = BaseAcquisition.from_config(
                source.config, model=model, best_f=source.best_f.detach().to(device), outcome_warp=warp
            )
    if isinstance(source, MACE):
        for name in ("qLogEI", "qPI", "qUCB"):
            getattr(acq, name).sampler.seed = getattr(source, name).sampler.seed
    return acq.to(device=device, dtype=torch.double)
