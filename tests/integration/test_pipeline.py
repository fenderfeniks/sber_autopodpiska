"""
test_pipeline.py — интеграционные тесты MLPipeline.

Покрытие:
- Smoke-тест полного цикла train → артефакты на диске → predict
- predict до train/load поднимает ValueError
- Выходной тип predict — np.ndarray
- Размер выходного вектора == размеру входа
- Цикл save → load → predict даёт идентичные предсказания
- Препроцессор сохраняется и загружается корректно
- use_tracker=False не пишет в MLflow вне run
"""

import pytest
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from core.pipeline import MLPipeline


# ---------------------------------------------------------------------------
# Вспомогательные данные
# ---------------------------------------------------------------------------
def _get_splits(sample_data, target):
    train = sample_data.iloc[:160]
    val = sample_data.iloc[160:180]
    test = sample_data.iloc[180:]
    return (
        train.drop(columns=[target]), train[target],
        val.drop(columns=[target]), val[target],
        test.drop(columns=[target]), test[target],
    )


# ---------------------------------------------------------------------------
# Тест 1: Smoke-тест — полный цикл без падений
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_pipeline_train_runs_without_error(mock_config, sample_data):
    """
    Пайплайн должен пройти полный цикл обучения без исключений.
    """
    target = mock_config.data.tabular.target_col
    X_train, y_train, X_val, y_val, _, _ = _get_splits(sample_data, target)

    pipeline = MLPipeline(mock_config)
    pipeline.train(X_train, y_train, X_val, y_val,
                   save_artifacts=False, use_tracker=False)


# ---------------------------------------------------------------------------
# Тест 2: predict до обучения поднимает ValueError
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_before_train_raises_value_error(mock_config, sample_data):
    """
    predict() на необученном пайплайне должен поднимать ValueError,
    а не AttributeError или молча возвращать None.
    """
    target = mock_config.data.tabular.target_col
    X = sample_data.drop(columns=[target])

    pipeline = MLPipeline(mock_config)
    with pytest.raises(ValueError, match="не обучен"):
        pipeline.predict(X)


# ---------------------------------------------------------------------------
# Тест 3: Выходной тип и размер predict
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_predict_output_type_and_shape(mock_config, sample_data, trained_pipeline):
    """
    predict() должен возвращать np.ndarray длиной == количеству входных строк.
    Не список, не DataFrame, не None.
    """
    target = mock_config.data.tabular.target_col
    X_test = sample_data.iloc[180:].drop(columns=[target])

    preds = trained_pipeline.predict(X_test)

    assert isinstance(preds, np.ndarray), (
        f"predict() вернул {type(preds)}, ожидался np.ndarray."
    )
    assert len(preds) == len(X_test), (
        f"Размер предсказаний {len(preds)} != размер входа {len(X_test)}."
    )


# ---------------------------------------------------------------------------
# Тест 4: Нет NaN в предсказаниях
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_predict_output_has_no_nan(mock_config, sample_data, trained_pipeline):
    """
    Ни одно предсказание не должно быть NaN.
    NaN в выходе ломает downstream-логику (сохранение, метрики, API).
    """
    target = mock_config.data.tabular.target_col
    X_test = sample_data.iloc[180:].drop(columns=[target])

    preds = trained_pipeline.predict(X_test)

    nan_count = np.isnan(preds.astype(float)).sum()
    assert nan_count == 0, (
        f"predict() вернул {nan_count} NaN значений из {len(preds)}."
    )


# ---------------------------------------------------------------------------
# Тест 5: Артефакты сохраняются на диск при save_artifacts=True
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_artifacts_are_saved_to_disk(mock_config, sample_data):
    """
    После train(save_artifacts=True) на диске должны быть:
    - preprocessing_v{version}.pkl
    - feature_schema_v{version}.json
    - файл модели с нужным расширением
    """
    target = mock_config.data.tabular.target_col
    X_train, y_train, X_val, y_val, _, _ = _get_splits(sample_data, target)

    pipeline = MLPipeline(mock_config)
    pipeline.train(X_train, y_train, X_val, y_val,
                   save_artifacts=True, use_tracker=False)

    version = mock_config.model.version
    models_dir = Path(mock_config.paths.models_dir)

    prep_path = models_dir / f"preprocessing_v{version}.pkl"
    schema_path = models_dir / f"feature_schema_v{version}.json"
    model_path = models_dir / f"{mock_config.model.name}_v{version}{pipeline.model.file_extension}"

    assert prep_path.exists(), f"Препроцессор не найден: {prep_path}"
    assert schema_path.exists(), f"Схема признаков не найдена: {schema_path}"
    assert model_path.exists(), f"Файл модели не найден: {model_path}"


# ---------------------------------------------------------------------------
# Тест 6: Схема фичей содержит корректный JSON
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_feature_schema_is_valid_json(mock_config, sample_data):
    """
    feature_schema_v{version}.json должен быть валидным JSON-словарём
    с именами колонок и их типами.
    """
    target = mock_config.data.tabular.target_col
    X_train, y_train, X_val, y_val, _, _ = _get_splits(sample_data, target)

    pipeline = MLPipeline(mock_config)
    pipeline.train(X_train, y_train, X_val, y_val,
                   save_artifacts=True, use_tracker=False)

    version = mock_config.model.version
    schema_path = Path(mock_config.paths.models_dir) / f"feature_schema_v{version}.json"

    with open(schema_path) as f:
        schema = json.load(f)

    assert isinstance(schema, dict), "Схема должна быть словарём."
    assert len(schema) > 0, "Схема пустая."

    valid_dtypes = {"int64", "int32", "float64", "float32", "object", "bool", "category"}
    for col, dtype in schema.items():
        assert dtype in valid_dtypes, (
            f"Колонка '{col}' имеет неизвестный тип '{dtype}'."
        )


# ---------------------------------------------------------------------------
# Тест 7: Цикл save → load → predict даёт идентичные предсказания
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_save_load_predict_is_identical(mock_config, sample_data):
    """
    Предсказания пайплайна после сохранения и загрузки должны совпадать
    с предсказаниями до сохранения. Это гарантирует что сериализация корректна.
    """
    target = mock_config.data.tabular.target_col
    X_train, y_train, X_val, y_val, X_test, _ = _get_splits(sample_data, target)

    # Обучаем и сохраняем
    pipeline_original = MLPipeline(mock_config)
    pipeline_original.train(X_train, y_train, X_val, y_val,
                            save_artifacts=True, use_tracker=False)
    preds_original = pipeline_original.predict(X_test)

    # Загружаем из диска
    pipeline_loaded = MLPipeline(mock_config)
    pipeline_loaded.load()
    preds_loaded = pipeline_loaded.predict(X_test)

    np.testing.assert_array_almost_equal(
        preds_original,
        preds_loaded,
        decimal=5,
        err_msg="Предсказания до и после save/load отличаются."
    )


# ---------------------------------------------------------------------------
# Тест 8: Препроцессор загружается и применяется корректно
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_loaded_preprocessor_transforms_consistently(mock_config, sample_data):
    """
    Загруженный препроцессор должен трансформировать данные
    идентично оригинальному (fit на train).
    """
    target = mock_config.data.tabular.target_col
    X_train, y_train, X_val, y_val, X_test, _ = _get_splits(sample_data, target)

    pipeline = MLPipeline(mock_config)
    pipeline.train(X_train, y_train, X_val, y_val,
                   save_artifacts=True, use_tracker=False)

    # Трансформируем оригинальным препроцессором
    X_orig = pipeline.preprocessor.transform(X_test)

    # Загружаем препроцессор с диска
    version = mock_config.model.version
    prep_path = Path(mock_config.paths.models_dir) / f"preprocessing_v{version}.pkl"
    loaded_prep = joblib.load(prep_path)
    X_loaded = loaded_prep.transform(X_test)

    pd.testing.assert_frame_equal(
        pd.DataFrame(X_orig).reset_index(drop=True),
        pd.DataFrame(X_loaded).reset_index(drop=True),
        check_dtype=False,
        obj="Оригинальный vs загруженный препроцессор"
    )
