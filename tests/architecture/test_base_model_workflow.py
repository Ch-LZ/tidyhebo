from dataclasses import replace
import pytest
import torch
import tidyhebo.base_model as module
from tidyhebo.base_model import SurrogateModel, SimpleModel
from tidyhebo.config import ModelConfig


def data():
    x = torch.linspace(0, 1, 6, dtype=torch.double)[:, None]
    return x, torch.sin(x * 4) + x


@pytest.mark.parametrize("target", ["joint", "centered", "profiled"])
def test_shared_training_uses_registered_pair_builder(monkeypatch, target):
    x, y = data()
    config = ModelConfig(_target_=target, input_warp=False, gate=False)
    model = SurrogateModel.from_config(config, train_x=x, train_y=y)
    calls = []

    def build():
        calls.append("build")
        return SimpleModel(x, y, config).build_gp_mll()

    def solve(mll, **kwargs):
        assert mll.model is model.model
        calls.append("fit")
        return mll

    monkeypatch.setattr(model, "build_gp_mll", build)
    monkeypatch.setattr(module, "fit_gpytorch_mll", solve)
    state = model.train_model()
    assert calls == ["build", "fit"]
    assert state.keys() == dict(model.model.named_parameters()).keys()
    assert torch.allclose(
        model.model.train_targets, model.model.outcome_warp_yj(y).squeeze(-1)
    )
    assert not model.model.training and model.model.prediction_strategy is None


@pytest.mark.parametrize("transform", ["standardize", "yeo_johnson"])
def test_warm_start_keeps_parameters_and_fresh_data_statistics(monkeypatch, transform):
    x, y = data()
    config = ModelConfig(input_warp=False, gate=False, output_transform=transform)
    old = SimpleModel(x, y, config)
    old.build_gp_mll()
    with torch.no_grad():
        old.model.mean_module.constant.fill_(1.25)
    legacy_state = {
        name: value.clone() for name, value in old.model.state_dict().items()
    }
    new = SimpleModel(x, y * 3, config)

    def solve(mll, **kwargs):
        assert mll.model.mean_module.constant.item() == 1.25
        return mll

    monkeypatch.setattr(module, "fit_gpytorch_mll", solve)
    state = new.train_model(legacy_state)
    if transform == "yeo_johnson":
        assert new.model.outcome_warp_yj._scale == (y * 3).std()
    else:
        assert torch.allclose(
            new.model.outcome_transform.stdvs, (y * 3).std().reshape(1, 1)
        )
    with torch.no_grad():
        new.model.mean_module.constant.fill_(5.0)
    assert state["mean_module.raw_constant"].item() == 1.25
    assert legacy_state["mean_module.raw_constant"].item() == 1.25


def test_gate_rebuilds_without_output_warp_and_keeps_config(monkeypatch):
    x, y = data()
    config = ModelConfig(input_warp=False)
    calls = []

    def solve(mll, **kwargs):
        calls.append(mll)
        if hasattr(mll.model, "outcome_warp_yj"):
            mll.model.outcome_warp_yj._scale = y.new_tensor(1e6)
        return mll

    monkeypatch.setattr(module, "fit_gpytorch_mll", solve)
    model = SimpleModel(x, y, config)
    model.train_model()
    assert len(calls) == 2 and calls[0].model is not calls[1].model
    assert not hasattr(model.model, "outcome_warp_yj")
    assert config.output_transform == "yeo_johnson"


def test_cold_fit_does_not_load_or_save_warm_state(monkeypatch):
    x, y = data()
    config = ModelConfig(input_warp=False, gate=False, warm_start=False)
    model = SimpleModel(x, y, config)
    monkeypatch.setattr(module, "fit_gpytorch_mll", lambda mll, **kwargs: mll)

    def forbidden(state):
        pytest.fail("Warm state must not be loaded during a cold fit")

    monkeypatch.setattr(model, "_load_model_state", forbidden)
    assert model.train_model({"bad": torch.ones(1)}) is None


def test_separate_warp_is_an_explicit_pending_feature():
    x, y = data()
    with pytest.raises(NotImplementedError, match="Separate output-warp"):
        SimpleModel(x, y, ModelConfig(warp_fit="separate")).train_model()
