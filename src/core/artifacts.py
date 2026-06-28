from __future__ import annotations

import logging
import mlflow
from typing import Dict, Any
from contextlib import contextmanager
from optuna_integration.mlflow import MLflowCallback
from core.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Единый менеджер для сохранения метрик, параметров и файлов (паттерн Facade)."""

    def __init__(self, cfg):
        self.cfg = cfg
        # 1. Подключаем базу SQLite для метрик и параметров
        self.tracking_uri = cfg.logging.mlflow.tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # 2. Собираем железный абсолютный URI-путь для папки с тяжелыми моделями
        rel_artifact_path = cfg.logging.mlflow.artifact_uri_rel
        self.artifact_uri = (PROJECT_ROOT / rel_artifact_path).as_uri()

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
        """Контекстный менеджер для управления жизненным циклом запуска с авто-версионированием."""
        with mlflow.start_run(run_name=run_name) as run:
            try:
                # 1. Вытаскиваем версии и чейнджлоги из наших новых конфигов
                tabular_cfg = self.cfg.get('data', {}).get('tabular', {})
                model_cfg = self.cfg.get('model', {})

                prep_version = tabular_cfg.get('preprocessing_version', '0.0.0')
                feat_version = tabular_cfg.get('features_version', '0.0.0')
                model_version = model_cfg.get('model_version', '0.0.0')

                prep_log = tabular_cfg.get('preprocessing_changelog', '')
                feat_log = tabular_cfg.get('features_changelog', '')
                model_log = model_cfg.get('model_changelog', '')

                # 2. Логируем версии как параметры (чтобы по ним можно было фильтровать в таблице MLflow)
                mlflow.log_params({
                    "version_preprocessing": prep_version,
                    "version_features": feat_version,
                    "version_model": model_version
                })

                # 3. Устанавливаем теги с описанием изменений (changelog улетает в метаданные рана)
                mlflow.set_tags({
                    "preprocessing_changelog": prep_log,
                    "features_changelog": feat_log,
                    "model_changelog": model_log,
                    "model_architecture": model_cfg.get('name', 'unknown')
                })
                
            except Exception as e:
                logger.warning(f"Не удалось автоматически записать метаданные версий в MLflow: {e}")

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
            return [MLflowCallback(tracking_uri=self.tracking_uri, metric_name=metric_name)]
        except ImportError:
            logger.warning("Пакет optuna_integration не установлен. Логирование Trials отключено.")
            return []