from dataclasses import replace

import pytest
import torch
import pandas as pd
import numpy as np

from tidyhebo.optimizer import tidyHEBO
from tidyhebo.config import DiscreteSearchConfig
from tidyhebo.mixed_params import RealP


@pytest.fixture
def tidyhebo_discrete():
    """Создаёт экземпляр tidyHEBO с небольшим набором данных и одним параметром."""
    param = RealP(0, name="x", lb=0.0, ub=1.0)
    train_x = pd.DataFrame({"x": [0.1, 0.3, 0.5, 0.7, 0.9]})
    train_y = np.array([0.2, 0.4, 0.6, 0.8, 0.9]).reshape(-1, 1)
    
    optimizer = tidyHEBO(
        parameters=[param],
        train_x=train_x,
        train_y=train_y,
    )

    return optimizer


def test_tidyhebo_discrete_suggest_single(tidyhebo_discrete):
    """Проверяет, что suggest() с DiscreteSearchConfig возвращает одну точку из множества кандидатов."""
    candidates = pd.DataFrame({"x": [0.2, 0.4, 0.6, 0.8]})
    
    tidyhebo_discrete.config = replace(tidyhebo_discrete.config,
        search=DiscreteSearchConfig(candidates=candidates), return_all=False)
    result = tidyhebo_discrete.suggest()
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1
    assert result.iloc[0]["x"] in candidates["x"].values


def test_tidyhebo_discrete_suggest_return_all(tidyhebo_discrete):
    """Проверяет, что config.return_all=True возвращает все точки Парето-фронта."""
    candidates = pd.DataFrame({"x": [0.2, 0.4, 0.6, 0.8]})
    
    tidyhebo_discrete.config = replace(tidyhebo_discrete.config,
        search=DiscreteSearchConfig(candidates=candidates), return_all=True)
    result = tidyhebo_discrete.suggest()
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] >= 1
    assert set(result["x"].values).issubset(set(candidates["x"].values))


def test_tidyhebo_discrete_suggest_fallback_to_continuous(tidyhebo_discrete):
    """Проверяет, что при GASearchConfig используется непрерывный метод (не падает)."""
    tidyhebo_discrete._init_design_generator = None
    result = tidyhebo_discrete.suggest()
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1
    assert 0.0 <= result.iloc[0]["x"] <= 1.0
