"""
test_latency.py — тесты производительности инференса.

Покрытие:
- Медианная латентность одного запроса укладывается в SLA
- P95 латентность (хвост распределения) не превышает 2x SLA
- Латентность батча из N строк масштабируется линейно (не экспоненциально)
- Повторные вызовы не деградируют (нет утечек памяти / state накопления)
"""

import pytest
import time
import numpy as np
import pandas as pd

# SLA-константы — централизованы здесь, не разбросаны по тестам
SINGLE_REQUEST_SLA_SEC = 0.050    # 50 мс — медианный SLA
P95_MULTIPLIER = 2.0              # P95 не должен превышать 2x SLA
WARMUP_RUNS = 5                   # прогрев JIT/кэша
MEASURE_RUNS = 30                 # количество замеров для статистики
BATCH_SIZES = [1, 10, 50, 100]    # размеры батчей для теста масштабируемости


def _measure_latencies(pipeline, X: pd.DataFrame, n_runs: int) -> np.ndarray:
    """Возвращает массив времён инференса в секундах."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        pipeline.predict(X)
        times.append(time.perf_counter() - start)
    return np.array(times)


# ---------------------------------------------------------------------------
# Тест 1: Медианная латентность одного запроса <= SLA
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
def test_single_request_median_latency_within_sla(mock_config, sample_data, trained_pipeline):
    """
    Медианная латентность инференса для одного запроса должна укладываться в SLA.

    Используем медиану (а не один замер) чтобы избежать флапания из-за
    GC-пауз, планировщика ОС, CPU throttling на CI.
    """
    target = mock_config.data.tabular.target_col
    single_row = sample_data.iloc[[0]].drop(columns=[target])

    # Прогрев: даём Python/JIT закэшировать библиотеки
    for _ in range(WARMUP_RUNS):
        trained_pipeline.predict(single_row)

    latencies = _measure_latencies(trained_pipeline, single_row, MEASURE_RUNS)
    median_latency = np.median(latencies)

    assert median_latency < SINGLE_REQUEST_SLA_SEC, (
        f"Медианная латентность {median_latency * 1000:.2f} мс превышает "
        f"SLA {SINGLE_REQUEST_SLA_SEC * 1000:.0f} мс.\n"
        f"Все замеры (мс): {(latencies * 1000).round(2).tolist()}"
    )


# ---------------------------------------------------------------------------
# Тест 2: P95 латентности не превышает 2x SLA
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
def test_single_request_p95_latency_within_limit(mock_config, sample_data, trained_pipeline):
    """
    95-й перцентиль латентности не должен превышать 2x SLA.
    Хвост распределения критичен для пользовательского опыта.
    """
    target = mock_config.data.tabular.target_col
    single_row = sample_data.iloc[[0]].drop(columns=[target])

    for _ in range(WARMUP_RUNS):
        trained_pipeline.predict(single_row)

    latencies = _measure_latencies(trained_pipeline, single_row, MEASURE_RUNS)
    p95_latency = np.percentile(latencies, 95)
    p95_limit = SINGLE_REQUEST_SLA_SEC * P95_MULTIPLIER

    assert p95_latency < p95_limit, (
        f"P95 латентность {p95_latency * 1000:.2f} мс превышает "
        f"лимит {p95_limit * 1000:.0f} мс (2x SLA).\n"
        f"P50={np.median(latencies)*1000:.2f} мс, "
        f"P95={p95_latency*1000:.2f} мс, "
        f"P99={np.percentile(latencies, 99)*1000:.2f} мс"
    )


# ---------------------------------------------------------------------------
# Тест 3: Латентность батча масштабируется субэкспоненциально
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
def test_batch_latency_scales_subexponentially(mock_config, sample_data, trained_pipeline):
    """
    Время инференса батча из N строк не должно расти быстрее чем O(N).

    Проверяем что латентность батча из 100 строк < 100 * латентность одной строки.
    Если это нарушается — есть O(N^2) операция (например построчный цикл).
    """
    target = mock_config.data.tabular.target_col

    results = {}
    for batch_size in BATCH_SIZES:
        X_batch = sample_data.iloc[:batch_size].drop(columns=[target])

        # Прогрев
        for _ in range(WARMUP_RUNS):
            trained_pipeline.predict(X_batch)

        latencies = _measure_latencies(trained_pipeline, X_batch, 10)
        results[batch_size] = np.median(latencies)

    # Латентность одной строки как baseline
    single_latency = results[1]

    for batch_size in BATCH_SIZES[1:]:
        batch_latency = results[batch_size]
        # Линейный верхний порог: batch не должен быть медленнее N одиночных запросов
        linear_limit = single_latency * batch_size

        assert batch_latency < linear_limit, (
            f"Батч из {batch_size} строк занял {batch_latency*1000:.2f} мс, "
            f"что медленнее {batch_size} одиночных запросов "
            f"({linear_limit*1000:.2f} мс). Возможна O(N^2) операция."
        )


# ---------------------------------------------------------------------------
# Тест 4: Повторные вызовы не деградируют (нет накопления состояния)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
def test_latency_does_not_degrade_over_time(mock_config, sample_data, trained_pipeline):
    """
    100-й вызов predict() не должен быть значительно медленнее 1-го.
    Деградация указывает на утечку памяти или накопление state внутри модели.

    Сравниваем медиану первых 10 замеров с медианой последних 10 из 100.
    """
    target = mock_config.data.tabular.target_col
    single_row = sample_data.iloc[[0]].drop(columns=[target])

    for _ in range(WARMUP_RUNS):
        trained_pipeline.predict(single_row)

    all_latencies = _measure_latencies(trained_pipeline, single_row, 100)

    early_median = np.median(all_latencies[:10])
    late_median = np.median(all_latencies[-10:])

    # Допускаем деградацию не более чем в 3 раза
    # (3x — очень щедрый порог, реальная деградация будет на порядки)
    degradation_ratio = late_median / (early_median + 1e-9)

    assert degradation_ratio < 3.0, (
        f"Латентность деградировала: первые 10 вызовов — {early_median*1000:.2f} мс, "
        f"последние 10 из 100 — {late_median*1000:.2f} мс "
        f"(ratio={degradation_ratio:.2f}x)."
    )
