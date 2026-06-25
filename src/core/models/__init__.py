from __future__ import annotations

from omegaconf import DictConfig
from .base import BaseModelWrapper
from .catboost import CatBoostWrapper
from .lightgbm import LightGBMWrapper
from .xgboost import XGBoostWrapper
from .pytorch import PyTorchWrapper

def get_model(config: DictConfig) -> BaseModelWrapper:
    model_name = config.model.name.lower()

    if "catboost" in model_name:
        return CatBoostWrapper(config)
    elif "lightgbm" in model_name or "lgb" in model_name:
        return LightGBMWrapper(config)
    elif "xgboost" in model_name or "xgb" in model_name:
        return XGBoostWrapper(config)
    elif "own_model" in model_name or "pytorch" in model_name:
        return PyTorchWrapper(config) # передаем только config, так как custom_nn опционален
    else:
        raise ValueError(f"Модель {model_name} не поддерживается фабрикой!")