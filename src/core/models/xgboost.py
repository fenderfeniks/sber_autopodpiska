from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf

from core.utils import PROJECT_ROOT
from .base import BaseModelWrapper

# Импортируем специфичные библиотеки внутри методов или через try/except,
# чтобы код не падал, если в проекте не установлен CatBoost или PyTorch
try:
    import xgboost as xgb

    XGBOOST_INSTALLED = True
except ImportError:
    XGBOOST_INSTALLED = False

logger = logging.getLogger(__name__)


# ============================================================
# 4. XGBoost
# ============================================================

class XGBoostWrapper(BaseModelWrapper):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        if not XGBOOST_INSTALLED:
            raise ImportError("Библиотека xgboost не установлена!")

        self.ml_cfg = self.cfg.training.ml
        full_params = OmegaConf.to_container(self.model_cfg.params, resolve=True)

        # Конфигурируем лосс, метрику и раннюю остановку
        full_params['objective'] = self.cfg.loss_function
        if self.cfg.metrics:
            # XGBoost ожидает строку или список строк в eval_metric
            full_params['eval_metric'] = self.cfg.metrics[0] if len(self.cfg.metrics) == 1 else self.cfg.metrics

        if self.ml_cfg.early_stopping_rounds > 0:
            full_params['early_stopping_rounds'] = self.ml_cfg.early_stopping_rounds

        # Динамическое переключение архитектуры под задачу
        if self.task_type == 'regression':
            self.model = xgb.XGBRegressor(**full_params)
        elif self.task_type in ['binary', 'multiclass']:
            self.model = xgb.XGBClassifier(**full_params)
        else:
            raise ValueError(f"Неизвестный task_type для XGBoost: {self.task_type}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
            tracker=None) -> None:  # <--- Добавлен tracker

        # ЛОГИРОВАНИЕ В TRACKER (Без прямого MLflow)
        if tracker:
            params_to_log = self.model.get_params()
            params_to_log.update({
                "model_name": self.model_cfg.name,
                "model_version": self.model_cfg.model_version
            })
            tracker.log_params(params_to_log)

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        verbose_val = self.ml_cfg.verbose if self.ml_cfg.verbose > 0 else False

        logger.info(f"Обучение XGBoost ({self.model_cfg.name})...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose_val
        )

        # Логирование лучшей метрики в МЕНЕДЖЕР
        if eval_set and hasattr(self.model, 'best_score') and tracker:
            tracker.log_metrics({"best_val_score": self.model.best_score})

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        raise NotImplementedError("predict_proba доступен только для задач классификации.")

    def save(self) -> str:
        """Нативное сохранение XGBoost в универсальный формат UBJSON."""
        file_name = f"{self.model_cfg.name}_v{self.model_cfg.model_version}.ubj"
        save_path = PROJECT_ROOT / self.cfg.paths.models_dir / file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(save_path))
        logger.info(f"Модель XGBoost нативно сохранена в {save_path}")

        # УДАЛЕНА ЛОГИКА MLFLOW ОТСЮДА

        return str(save_path)

    def load(self, load_path: str) -> None:
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Файл модели XGBoost не найден: {load_path}")

        # Метод load_model корректно восстанавливает внутреннее состояние нужного класса
        self.model.load_model(str(load_path))
        logger.info(f"Модель XGBoost успешно загружена из {load_path}")

    @property
    def file_extension(self) -> str:
        return ".ubj"

    def get_best_val_score(self, metric_name: str) -> float:
        # XGBoost sklearn API хранит только одну метрику (первую из eval_metric).
        # metric_name игнорируется — убедись, что в конфиге metrics[0] совпадает с eval_metric.
        if hasattr(self.model, 'best_score'):
            return self.model.best_score
        return 0.0