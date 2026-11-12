from dataclasses import FrozenInstanceError, dataclass, replace
import pandas as pd
import pytest
from tidyhebo.registry import Registrable
from tidyhebo.config import (
    TidyHEBOConfig,
    SearchConfig,
    GASearchConfig,
    DiscreteSearchConfig,
    BoTorchSearchConfig,
)


def test_registry_families_extensions_and_constructor_errors():
    class First(Registrable):
        pass

    class Second(Registrable):
        pass

    class A(First, _target_="same"):
        def __init__(self, config, dependency):
            self.dependency = dependency

    class B(Second, _target_="same"):
        def __init__(self, config):
            pass

    @dataclass
    class Config:
        _target_: str = "same"

    value = object()
    assert First.from_config(Config(), dependency=value).dependency is value
    assert isinstance(Second.from_config(Config()), B)
    with pytest.raises(ValueError, match="already registered"):
        class Duplicate(First, _target_="same"):
            pass
    with pytest.raises(ValueError, match="Unknown"):
        First.from_config(Config("missing"))

    class Broken(First, _target_="broken"):
        def __init__(self, config):
            raise KeyError("inside constructor")

    with pytest.raises(KeyError, match="inside constructor"):
        First.from_config(Config("broken"))


def test_configs_are_independent_immutable_data():
    first, second = TidyHEBOConfig(), TidyHEBOConfig()
    assert first.model is not second.model
    assert first.search.stop is not second.search.stop
    changed = replace(first, model=replace(first.model, _target_="centered"))
    assert first.model._target_ == "joint"
    assert changed.model._target_ == "centered"
    with pytest.raises(FrozenInstanceError):
        first.acquisition.q = 8


def test_search_configs_share_one_base():
    for config in (
        GASearchConfig(),
        DiscreteSearchConfig(candidates=pd.DataFrame({"x": [0.5]})),
        BoTorchSearchConfig(),
    ):
        assert isinstance(config, SearchConfig)
        assert replace(TidyHEBOConfig(), search=config).search is config
    assert isinstance(TidyHEBOConfig().search, GASearchConfig)
