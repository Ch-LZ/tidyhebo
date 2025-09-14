from tidyhebo.common import dtype
import torch


def test_dtype():
    assert type(dtype) == type(torch.double)
