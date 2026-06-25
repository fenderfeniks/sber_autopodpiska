"""
conftest.py — базовые фикстуры для всех тестов.

Ключевые принципы:
- mock_config имеет scope="function": каждый тест получает свежую копию,
  мутации одного теста не влияют на другие.
- sample_ga_data генерируется через np.random.Generator (не global seed),
  что делает генерацию детерминированной и изолированной.
- pipeline_factory — фабрика, возвращающая обученный пайплайн;
  используется в нескольких тестах без дублирования логики обучения.
"""

import pytest
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize, compose
import mlflow


# ---------------------------------------------------------------------------
# Константы тестового датасета
# ---------------------------------------------------------------------------
N_ROWS = 200          # достаточно для стратификации и надёжных сплитов
RANDOM_SEED = 42
TARGET_COL = "is_target_action"
INFERENCE_SLA_SEC = 0.050   # 50 мс — SLA на один запрос
LATENCY_WARMUP_RUNS = 3
LATENCY_MEASURE_RUNS = 20


# ---------------------------------------------------------------------------
# Фикстура конфига — scope="function" чтобы мутации не просачивались
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def mock_config(tmp_path):
    """
    Создаёт изолированный конфиг для каждого теста.
    - MLflow пишет в SQLite во временную папку (не засоряет рабочую среду).
    - models_dir изолирован в tmp_path.
    - Struct mode снят: тесты могут переопределять поля.
    """
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.tabular.target_col={TARGET_COL}",
                "env=dev",
                "data.sample_pct=1.0",
                "data.test_size=0.2",
                "data.val_size=0.2",
            ]
        )

    OmegaConf.set_struct(cfg, False)

    # Изолируем пути и MLflow в tmp_path конкретного теста
    cfg.data_dir = str(tmp_path / "data")

    cfg.paths.data_dir = str(tmp_path / "data")
    cfg.paths.raw_dir = str(tmp_path / "data/raw")
    cfg.paths.processed_dir = str(tmp_path / "data/processed")
    cfg.paths.features_dir = str(tmp_path / "data/features")  # <--- Не хватало этого
    cfg.paths.logs_dir = str(tmp_path / "logs")
    cfg.paths.models_dir = str(tmp_path / "models")

    cfg.logging.mlflow.tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    cfg.logging.log_file = str(tmp_path / "test_pipeline.log")

    Path(cfg.paths.models_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================
    # ИСПРАВЛЕНИЕ: Жестко сбрасываем кэш MLflow для нового теста
    # =========================================================
    mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
    mlflow.set_experiment("Default")
    # =========================================================

    return cfg


# ---------------------------------------------------------------------------
# Синтетический датасет — детерминированный, изолированный
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def sample_data():
    """
    Генерирует синтетический датасет с контролируемыми свойствами:
    - Фиксированный seed через Generator (не влияет на global numpy state).
    - Явный дисбаланс классов (80/20) — реалистичен для бизнес-задач.
    - NaN в категориальных признаках (~5%) — проверяет robustness.
    - Константная колонка — проверяет что препроцессор её дропает.
    - Высококоррелированная колонка — проверяет что модель не падает.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = N_ROWS

    df = pd.DataFrame({
        "session_id": [f"sess_{i}" for i in range(n)],
        "visitStartTime": rng.integers(1_600_000_000, 1_610_000_000, size=n),
        "device_category": rng.choice(["mobile", "desktop", "tablet"], size=n),
        "browser": rng.choice(["Chrome", "Safari", "Firefox"], size=n),
        "total_hits": rng.integers(1, 50, size=n),
        "visit_number": rng.integers(1, 10, size=n),
        "constant_col": 42,                         # константная колонка
        TARGET_COL: rng.choice([0, 1], size=n, p=[0.8, 0.2]),
    })

    # ~5% NaN в браузере
    nan_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[nan_idx, "browser"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Фабрика обученного пайплайна — переиспользуется в нескольких файлах
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def trained_pipeline(mock_config, sample_data):
    """
    Возвращает уже обученный MLPipeline.
    Используется в тестах где нужен инференс, а не само обучение.
    """
    from core.pipeline import MLPipeline

    target = mock_config.data.tabular.target_col
    train_df = sample_data.iloc[:160]
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_val = sample_data.iloc[160:180].drop(columns=[target])
    y_val = sample_data.iloc[160:180][target]

    pipeline = MLPipeline(mock_config)
    pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=False, use_tracker=False)
    return pipeline

@pytest.fixture(scope="function", autouse=True)
def cleanup_mlflow():
    """Гарантирует, что ни один тест не оставит за собой открытый MLflow run."""
    yield
    while mlflow.active_run():
        mlflow.end_run()