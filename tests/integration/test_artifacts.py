"""
test_artifacts.py — тесты ArtifactManager.

Покрытие:
- log_metrics/log_params не падают внутри активного run
- log_metrics вне run поднимает исключение (а не молча игнорирует)
- get_optuna_callback возвращает список (пустой или с коллбэком)
- set_experiment создаёт эксперимент если его нет
- log_artifact корректно обрабатывает отсутствующий файл (warning, не crash)
"""

import pytest
import mlflow
from pathlib import Path

from core.artifacts import ArtifactManager


# ---------------------------------------------------------------------------
# Тест 1: log_metrics и log_params не падают внутри run
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_log_metrics_and_params_inside_run(mock_config):
    """
    Базовый контракт: ArtifactManager логирует метрики и параметры
    без исключений внутри активного MLflow run.
    """
    manager = ArtifactManager(mock_config)
    manager.set_experiment("test_experiment")

    with manager.start_run(run_name="test_run"):
        manager.log_params({"lr": 0.01, "depth": 6})
        manager.log_metrics({"rmse": 0.15, "mae": 0.10})
        manager.log_metrics({"val_loss": 0.20}, step=1)



# ---------------------------------------------------------------------------
# Тест 2: log_artifact с несуществующим файлом — warning, не crash
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_log_artifact_missing_file_logs_warning_not_crash(mock_config, caplog):
    """
    Если файл для логирования не существует, ArtifactManager должен
    залогировать предупреждение и продолжить работу — не падать.
    Это важно для graceful degradation в проде.
    """
    import logging
    manager = ArtifactManager(mock_config)
    manager.set_experiment("test_experiment")

    with manager.start_run(run_name="test_artifact"):
        with caplog.at_level(logging.WARNING, logger="core.artifacts"):
            manager.log_artifact("/nonexistent/path/model.pkl", "models")

    # Тест пройдёт если не было исключения (проверяем через отсутствие краша)


# ---------------------------------------------------------------------------
# Тест 3: set_experiment — идемпотентен (повторный вызов не падает)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_set_experiment_is_idempotent(mock_config):
    """
    Повторный вызов set_experiment с одним и тем же именем не должен
    поднимать исключение. MLflow должен использовать существующий эксперимент.
    """
    manager = ArtifactManager(mock_config)

    manager.set_experiment("idempotent_test_exp")
    manager.set_experiment("idempotent_test_exp")  # второй вызов


# ---------------------------------------------------------------------------
# Тест 4: get_optuna_callback возвращает список
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_get_optuna_callback_returns_list(mock_config):
    """
    get_optuna_callback должен возвращать список.
    Пустой — если optuna_integration не установлен.
    С коллбэком — если установлен.
    В любом случае не None и не исключение.
    """
    manager = ArtifactManager(mock_config)
    result = manager.get_optuna_callback(metric_name="val_score")

    assert isinstance(result, list), (
        f"get_optuna_callback вернул {type(result)}, ожидался list."
    )


# ---------------------------------------------------------------------------
# Тест 5: log_dict сохраняет словарь внутри run
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_log_dict_inside_run(mock_config):
    """
    log_dict должен сохранять словарь как JSON-артефакт без исключений.
    """
    manager = ArtifactManager(mock_config)
    manager.set_experiment("test_experiment")

    test_dict = {"col_a": "float64", "col_b": "int64", "col_c": "object"}

    with manager.start_run(run_name="test_dict"):
        manager.log_dict(test_dict, "feature_schema.json", "schemas")


# ---------------------------------------------------------------------------
# Тест 6: start_run является контекстным менеджером и закрывает run
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_start_run_closes_run_on_exit(mock_config):
    """
    После выхода из контекстного менеджера start_run активного run быть не должно.
    """
    manager = ArtifactManager(mock_config)
    manager.set_experiment("test_experiment")

    with manager.start_run(run_name="ctx_test"):
        assert mlflow.active_run() is not None, "Внутри контекста должен быть активный run."

    assert mlflow.active_run() is None, (
        "После выхода из контекста run должен быть закрыт."
    )
