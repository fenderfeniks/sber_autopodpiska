from __future__ import annotations

import logging
import mlflow
from typing import Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Единый менеджер для сохранения метрик, параметров и файлов (паттерн Facade)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.tracking_uri = cfg.logging.mlflow.tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)

    def set_experiment(self, experiment_name: str, artifact_location: str = None):
        """Устанавливает или создает эксперимент с привязкой к локальной папке."""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location
            )
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str = None):
        """Контекстный менеджер для управления жизненным циклом запуска."""
        with mlflow.start_run(run_name=run_name) as run:
            yield run

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Сохраняет словарь метрик."""
        for k, v in metrics.items():
            # Санируем ключ: заменяем двоеточия, знаки равенства и запятые на '_'
            safe_key = (
                k.replace(":", "_")
                 .replace("=", "_")
                 .replace(",", "_")
            )
            mlflow.log_metric(safe_key, v, step=step)

    def log_params(self, params: Dict[str, Any]):
        """Сохраняет гиперпараметры."""
        mlflow.log_params(params)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Загружает локальный файл (веса, препроцессор) в хранилище трекера."""
        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            logger.debug(f"Артефакт {local_path} сохранен в {artifact_path}")
        except Exception as e:
            logger.warning(f"Не удалось отправить артефакт в трекер: {e}")

    def log_dict(self, dictionary: dict, file_name: str, artifact_path: str = None):
        """Сохраняет словарь (например, схему) напрямую как JSON в трекер."""
        full_path = f"{artifact_path}/{file_name}" if artifact_path else file_name
        try:
            mlflow.log_dict(dictionary, full_path)
        except Exception as e:
            logger.warning(f"Не удалось сохранить словарь {file_name} в трекер: {e}")

    def get_optuna_callback(self, metric_name: str):
        """Фабрика коллбэков для Optuna (скрывает MLflow под капотом)."""
        try:
            from optuna_integration.mlflow import MLflowCallback
            return [MLflowCallback(tracking_uri=self.tracking_uri, metric_name=metric_name)]
        except ImportError:
            logger.warning("Пакет optuna_integration не установлен. Логирование Trials отключено.")
            return []