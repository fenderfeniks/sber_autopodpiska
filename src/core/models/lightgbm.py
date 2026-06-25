from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import joblib
from omegaconf import DictConfig, OmegaConf

from core.utils import PROJECT_ROOT
from .base import BaseModelWrapper

# Импортируем специфичные библиотеки внутри методов или через try/except,
# чтобы код не падал, если в проекте не установлен CatBoost или PyTorch

try:
    import lightgbm as lgb

    LIGHTGBM_INSTALLED = True
except ImportError:
    LIGHTGBM_INSTALLED = False

logger = logging.getLogger(__name__)


# ============================================================
# 3. LightGBM
# ============================================================

class LightGBMWrapper(BaseModelWrapper):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        if not LIGHTGBM_INSTALLED:
            raise ImportError("Библиотека lightgbm не установлена!")

        self.ml_cfg = self.cfg.training.ml
        full_params = OmegaConf.to_container(self.model_cfg.params, resolve=True)

        # Передаем метрику и лосс из корня конфигурации
        full_params['objective'] = self.cfg.loss_function
        if self.cfg.metrics:
            full_params['metric'] = list(self.cfg.metrics)

        # Динамическое переключение архитектуры под задачу
        if self.task_type == 'regression':
            self.model = lgb.LGBMRegressor(**full_params)
        elif self.task_type in ['binary', 'multiclass']:
            self.model = lgb.LGBMClassifier(**full_params)
        else:
            raise ValueError(f"Неизвестный task_type для LightGBM: {self.task_type}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
            tracker=None) -> None:  # <--- Добавлен tracker

        # ЛОГИРОВАНИЕ В TRACKER (Без прямого MLflow)
        if tracker:
            params_to_log = self.model.get_params()
            params_to_log.update({
                "model_name": self.model_cfg.name,
                "model_version": self.model_cfg.version
            })
            tracker.log_params(params_to_log)

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

        # Настройка ранней остановки через коллбэки
        callbacks = []
        if eval_set and self.ml_cfg.early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=self.ml_cfg.early_stopping_rounds, verbose=False))
        if self.ml_cfg.verbose > 0:
            callbacks.append(lgb.log_evaluation(period=self.ml_cfg.verbose))

        logger.info(f"Обучение LightGBM ({self.model_cfg.name})...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

        # Логирование метрик валидации в МЕНЕДЖЕР
        if eval_set and hasattr(self.model, 'best_score_') and tracker:
            metrics_to_log = {f"best_val_{metric_name}": score for metric_name, score in
                              self.model.best_score_['valid_0'].items()}
            tracker.log_metrics(metrics_to_log)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        raise NotImplementedError("predict_proba доступен только для задач классификации.")

    def save(self) -> str:
        """Сохранение Scikit-Learn интерфейса LightGBM через joblib."""
        file_name = f"{self.model_cfg.name}_v{self.model_cfg.version}.pkl"
        save_path = PROJECT_ROOT / self.cfg.paths.models_dir / file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, save_path)
        logger.info(f"Интерфейс LightGBM сохранен в {save_path}")

        # УДАЛЕНА ЛОГИКА MLFLOW ОТСЮДА

        return str(save_path)

    def load(self, load_path: str) -> None:
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Файл модели LightGBM не найден: {load_path}")
        self.model = joblib.load(load_path)
        logger.info(f"Модель LightGBM успешно загружена из {load_path}")

    @property
    def file_extension(self) -> str:
        return ".pkl"

    def get_best_val_score(self, metric_name: str) -> float:
        if not hasattr(self.model, 'best_score_') or 'valid_0' not in self.model.best_score_:
            return 0.0
        val_scores = self.model.best_score_['valid_0']
        for key, value in val_scores.items():
            if key.lower() == metric_name.lower():
                return value
        logger.warning(f"Метрика '{metric_name}' не найдена в {list(val_scores.keys())}. Берём первую.")
        return next(iter(val_scores.values()), 0.0)