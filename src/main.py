from __future__ import annotations

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd

# Импортируем только наши ООП-компоненты из ядра
from core.data import UniversalDataLoader
from core.pipeline import MLPipeline
from core.tuner import OptunaTuner
from core.utils import PROJECT_ROOT
from core.artifacts import ArtifactManager
from core.metrics import calculate_metrics

from hydra.core.config_store import ConfigStore
from core.config_schema import AppConfig

cs = ConfigStore.instance()
cs.store(name="config", node=AppConfig)

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig):
    """Настройка динамического логирования в файл на основе конфига Hydra."""
    log_file_path = PROJECT_ROOT / cfg.logging.log_file
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()

    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path) for h in root_logger.handlers):
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root_logger.addHandler(file_handler)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    mode = cfg.mode.lower()
    if mode == "eval": mode = "evaluate"

    # 1. Инициализируем логирование в файл /logs/pipeline.log
    setup_logging(cfg)

    logger.info(f"=== ЗАПУСК ORCHESTRATOR | РЕЖИМ: {mode.upper()} ===")

    # 2. НАСТРОЙКА ИНФРАСТРУКТУРЫ АРТЕФАКТОВ (Без прямого вызова MLflow)
    tracker = ArtifactManager(cfg)
    experiment_name = cfg.logging.mlflow.experiments.get(mode, "default_experiment")

    # Явно указываем локальный путь для тяжелых артефактов и передаем трекеру
    local_artifact_repo = str(PROJECT_ROOT / cfg.paths.logs_dir / "mlruns")
    tracker.set_experiment(experiment_name, artifact_location=local_artifact_repo)

    # 3. Загрузка данных (Общая для всех режимов)
    loader = UniversalDataLoader(cfg)
    df = loader.load_data()
    target = cfg.data.tabular.target_col

    # ==========================================================
    # РЕЖИМ 1: ТРЕНИРОВКА
    # ==========================================================
    if mode == "eda":
        logger.info("Запуск модуля EDA...")
        # run_eda(cfg)
    elif mode == "train":
        train_df, val_df, _ = loader.get_splits(df)

        X_train, y_train = train_df.drop(columns=[target]), train_df[target]
        X_val, y_val = val_df.drop(columns=[target]), val_df[target]

        # Запускаем через контекстный менеджер трекера
        with tracker.start_run(run_name=cfg.run_name):
            pipeline = MLPipeline(cfg)
            pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=True)

    # ==========================================================
    # РЕЖИМ 2: ТЮНИНГ (OPTUNA)
    # ==========================================================
    elif mode == "tune":
        train_df, val_df, _ = loader.get_splits(df)

        X_train, y_train = train_df.drop(columns=[target]), train_df[target]
        X_val, y_val = val_df.drop(columns=[target]), val_df[target]

        with tracker.start_run(run_name=f"{cfg.run_name}_optuna"):
            tuner = OptunaTuner(cfg)
            best_params = tuner.tune(X_train, y_train, X_val, y_val, tracker=tracker)

            logger.info("Тюнинг завершен. Обучение финальной модели...")
            for key, value in best_params.items():
                OmegaConf.update(cfg, f"model.params.{key}", value, merge=True)

            # Передаем эстафету пайплайну
            pipeline = MLPipeline(cfg)
            pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=True)

    # ==========================================================
    # РЕЖИМ 3: ОЦЕНКА (EVALUATE) - Тестирование отложенной выборки
    # ==========================================================
    elif mode == "evaluate":
        logger.info("Запуск оценки модели на тестовой выборке...")
        # Предполагаем, что get_splits возвращает Train, Val, Test. Нам нужен Test.
        _, _, test_df = loader.get_splits(df)

        if test_df is None or test_df.empty:
            raise ValueError("Тестовая выборка пуста! Проверьте логику DataLoader.")

        X_test, y_test = test_df.drop(columns=[target]), test_df[target]

        with tracker.start_run(run_name=f"{cfg.model.name}_v{cfg.model.version}_eval"):
            # Восстанавливаем пайплайн
            pipeline = MLPipeline(cfg)
            pipeline.load()

            # Делаем предсказания
            X_test_clean = pipeline.preprocessor.transform(X_test)
            y_pred = pipeline.model.predict(X_test_clean)

            y_prob = None
            if cfg.task_type in ['binary', 'multiclass']:
                try:
                    if hasattr(pipeline.model, 'predict_proba'):
                        y_prob = pipeline.model.predict_proba(X_test_clean)
                except NotImplementedError:
                    pass

            # Считаем и логируем метрик
            test_metrics = calculate_metrics(y_test, y_pred, cfg.task_type, y_prob)

            metrics_to_log = {f"test_{k}": v for k, v in test_metrics.items()}
            tracker.log_metrics(metrics_to_log)

            logger.info("=== РЕЗУЛЬТАТЫ EVALUATION ===")
            for m_name, m_value in test_metrics.items():
                logger.info(f"Test {m_name.upper()}: {m_value:.4f}")

    # ==========================================================
    # РЕЖИМ 4: ИНФЕРЕНС (INFERENCE) - Прогноз новых (слепых) данных
    # ==========================================================
    elif mode == "inference":
        logger.info("Запуск инференса на новых данных...")

        # Для инференса таргета может не быть в датасете
        X_new = df.drop(columns=[target]) if target in df.columns else df

        pipeline = MLPipeline(cfg)
        pipeline.load()

        # Сквозной предикт (очистка + прогноз)
        predictions = pipeline.predict(X_new)

        # Формируем итоговый DataFrame с результатами
        # (Если в данных был ID, его желательно оставить, чтобы заказчик понял, где чья строка)
        result_df = pd.DataFrame({"prediction": predictions})

        # Если это классификация, добавляем вероятности
        if cfg.task_type in ['binary', 'multiclass']:
            X_new_clean = pipeline.preprocessor.transform(X_new)
            try:
                if hasattr(pipeline.model, 'predict_proba'):
                    probs = pipeline.model.predict_proba(X_new_clean)
                    if cfg.task_type == 'binary':
                        result_df["probability"] = probs[:, 1]
            except NotImplementedError:
                pass

        # Сохраняем в CSV
        output_path = PROJECT_ROOT / cfg.paths.data_dir / f"predictions_{cfg.model.name}_v{cfg.model.version}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)

        logger.info(f"Инференс успешно завершен! Результаты сохранены в: {output_path}")

    else:
        raise ValueError(f"Неизвестный режим: {mode}")


if __name__ == "__main__":
    main()