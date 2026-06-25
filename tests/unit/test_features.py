"""
test_features.py — юнит-тесты для TabularPreprocessor.

Покрытие:
- sklearn-контракт: transform до fit бросает NotFittedError
- Количество строк не изменяется
- NaN полностью устраняются
- drop_cols из конфига действительно удаляются
- Константные колонки удаляются автоматически
- Статистики считаются только по train (no leakage в fit)
- transform на test не использует test-статистики
- Выбросы обрезаются корректно (zscore)
- fit_transform == fit + transform (идемпотентность)
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.utils.validation import NotFittedError
from omegaconf import OmegaConf

from core.features import TabularPreprocessor


# ---------------------------------------------------------------------------
# Вспомогательные данные
# ---------------------------------------------------------------------------
@pytest.fixture
def X_train_clean(sample_data, mock_config):
    target = mock_config.data.tabular.target_col
    return sample_data.iloc[:160].drop(columns=[target]).copy()


@pytest.fixture
def X_test_clean(sample_data, mock_config):
    target = mock_config.data.tabular.target_col
    return sample_data.iloc[160:].drop(columns=[target]).copy()


# ---------------------------------------------------------------------------
# Тест 1: sklearn-контракт — transform до fit бросает NotFittedError
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_transform_before_fit_raises_not_fitted_error(mock_config, X_train_clean):
    """
    Вызов transform() до fit() должен поднимать NotFittedError.
    Это стандартный sklearn-контракт.
    """
    preprocessor = TabularPreprocessor(mock_config)

    with pytest.raises(NotFittedError):
        preprocessor.transform(X_train_clean)


# ---------------------------------------------------------------------------
# Тест 2: Количество строк не изменяется
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_row_count_preserved_after_transform(mock_config, X_train_clean):
    """
    Препроцессор не должен удалять строки — это делает DataLoader отдельно.
    """
    preprocessor = TabularPreprocessor(mock_config)
    X_out = preprocessor.fit_transform(X_train_clean)

    assert X_out.shape[0] == X_train_clean.shape[0], (
        f"Строк на входе: {X_train_clean.shape[0]}, на выходе: {X_out.shape[0]}."
    )


# ---------------------------------------------------------------------------
# Тест 3: После transform не должно остаться NaN
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_nan_after_transform(mock_config, X_train_clean):
    """
    После препроцессинга ни одного NaN не должно остаться.
    Иначе CatBoost/XGBoost/sklearn модели могут упасть или дать неверный результат.
    """
    preprocessor = TabularPreprocessor(mock_config)
    X_out = preprocessor.fit_transform(X_train_clean)

    nan_counts = pd.DataFrame(X_out).isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0].to_dict()

    assert len(cols_with_nan) == 0, (
        f"После препроцессинга остались NaN: {cols_with_nan}."
    )


# ---------------------------------------------------------------------------
# Тест 4: Явные drop_cols из конфига действительно удаляются
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_explicit_drop_cols_are_removed(mock_config, X_train_clean):
    """
    Колонки из cfg.data.tabular.drop_cols должны отсутствовать в выходе.
    """
    # Добавляем колонку которую явно просим дропнуть
    X_with_id = X_train_clean.copy()
    X_with_id["session_id_to_drop"] = "some_value"

    from omegaconf import OmegaConf
    OmegaConf.update(mock_config, "data.tabular.drop_cols", ["session_id_to_drop"])

    preprocessor = TabularPreprocessor(mock_config)
    X_out = preprocessor.fit_transform(X_with_id)

    assert "session_id_to_drop" not in pd.DataFrame(X_out).columns, (
        "Колонка из drop_cols не была удалена препроцессором."
    )


# ---------------------------------------------------------------------------
# Тест 5: Константная колонка удаляется автоматически
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_constant_column_is_auto_dropped(mock_config, X_train_clean):
    """
    Колонки с одним уникальным значением (константы) не несут информации
    и должны удаляться автоматически в fit().
    """
    X_with_const = X_train_clean.copy()
    X_with_const["always_zero"] = 0

    preprocessor = TabularPreprocessor(mock_config)
    X_out = preprocessor.fit_transform(X_with_const)

    assert "always_zero" not in pd.DataFrame(X_out).columns, (
        "Константная колонка не была удалена автоматически."
    )


# ---------------------------------------------------------------------------
# Тест 6: Статистики fit считаются только по train (no leakage)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_fit_statistics_use_only_train_data(mock_config, X_train_clean, X_test_clean):
    """
    fill_values_ должны соответствовать статистикам train, а не test.
    Это ключевой тест на отсутствие data leakage в препроцессоре.
    """
    # ОТКЛЮЧАЕМ УДАЛЕНИЕ КОНСТАНТ ДЛЯ ЭТОГО ТЕСТА
    OmegaConf.update(mock_config, "data.tabular.max_constant_pct", 1.0)
    # Делаем train и test заведомо разными по median числовой колонки
    X_train_mod = X_train_clean.copy()
    X_test_mod = X_test_clean.copy()

    X_train_mod["total_hits"] = 5    # медиана train = 5
    X_test_mod["total_hits"] = 999   # медиана test = 999

    # Добавляем NaN в обе выборки чтобы fill_values_ применился
    X_train_mod.loc[0, "total_hits"] = np.nan
    X_test_mod.loc[X_test_mod.index[0], "total_hits"] = np.nan

    preprocessor = TabularPreprocessor(mock_config)
    preprocessor.fit(X_train_mod)

    # fill_values для total_hits должно быть близко к 5 (медиана train),
    # а не к 999 (медиана test)
    fill_val = preprocessor.fill_values_.get("total_hits", None)
    assert fill_val is not None, "fill_values_ не содержит 'total_hits'."
    assert fill_val < 100, (
        f"fill_values_['total_hits'] = {fill_val}. "
        f"Похоже что статистика считалась по test (999), а не по train (5)."
    )


# ---------------------------------------------------------------------------
# Тест 7: transform на test использует статистики train
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_transform_applies_train_statistics_to_test(mock_config, X_train_clean, X_test_clean):
    """
    После fit на train, transform на test должен заполнять NaN
    значением из train-статистики, а не из test.
    """
    preprocessor = TabularPreprocessor(mock_config)
    preprocessor.fit(X_train_clean)

    # Полностью обнуляем числовую колонку в test
    X_test_all_nan = X_test_clean.copy()
    X_test_all_nan["total_hits"] = np.nan

    X_out = preprocessor.transform(X_test_all_nan)
    result_df = pd.DataFrame(X_out)

    if "total_hits" in result_df.columns:
        assert result_df["total_hits"].isnull().sum() == 0, (
            "После transform на test остались NaN в total_hits."
        )
        # Значение должно быть fill_value из train, не из test
        expected_fill = preprocessor.fill_values_.get("total_hits")
        assert (result_df["total_hits"] == expected_fill).all(), (
            f"NaN заполнены неверным значением. "
            f"Ожидалось {expected_fill} (из train)."
        )


# ---------------------------------------------------------------------------
# Тест 8: Выбросы обрезаются при outlier_method=zscore
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_outlier_clipping_with_zscore(mock_config, X_train_clean):
    """
    При outlier_method=zscore значения за порогом должны быть обрезаны.
    После clipping не должно быть значений за пределами [mean ± thresh*std].
    """
    from omegaconf import OmegaConf
    OmegaConf.update(mock_config, "data.tabular.outlier_method", "zscore")
    OmegaConf.update(mock_config, "data.tabular.outlier_threshold", 2.0)

    X_with_outliers = X_train_clean.copy()
    # Добавляем явный выброс
    X_with_outliers.loc[0, "total_hits"] = 999_999

    preprocessor = TabularPreprocessor(mock_config)
    X_out = preprocessor.fit_transform(X_with_outliers)
    result_df = pd.DataFrame(X_out)

    if "total_hits" in result_df.columns and "total_hits" in preprocessor.outlier_bounds_:
        lower, upper = preprocessor.outlier_bounds_["total_hits"]
        assert result_df["total_hits"].max() <= upper + 1e-6, (
            f"После zscore clipping есть значение выше верхней границы {upper}."
        )
        assert result_df["total_hits"].min() >= lower - 1e-6, (
            f"После zscore clipping есть значение ниже нижней границы {lower}."
        )


# ---------------------------------------------------------------------------
# Тест 9: fit_transform == fit + transform (идемпотентность)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_fit_transform_equals_fit_then_transform(mock_config, X_train_clean):
    """
    fit_transform(X) должен давать тот же результат что fit(X) + transform(X).
    Это контракт sklearn TransformerMixin.
    """
    prep_1 = TabularPreprocessor(mock_config)
    out_1 = prep_1.fit_transform(X_train_clean)

    prep_2 = TabularPreprocessor(mock_config)
    prep_2.fit(X_train_clean)
    out_2 = prep_2.transform(X_train_clean)

    pd.testing.assert_frame_equal(
        pd.DataFrame(out_1).reset_index(drop=True),
        pd.DataFrame(out_2).reset_index(drop=True),
        check_dtype=False,
        obj="fit_transform vs fit+transform"
    )


# ---------------------------------------------------------------------------
# Тест 10: learned_drop_cols_ включает явные и автоматические колонки
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_learned_drop_cols_contains_all_bad_columns(mock_config, X_train_clean):
    """
    learned_drop_cols_ после fit должен содержать:
    1. Колонки из drop_cols (явные)
    2. Константные колонки (автоматические)
    """
    from omegaconf import OmegaConf

    X = X_train_clean.copy()
    X["explicit_drop"] = "keep_me_not"
    X["auto_const"] = 1

    OmegaConf.update(mock_config, "data.tabular.drop_cols", ["explicit_drop"])

    preprocessor = TabularPreprocessor(mock_config)
    preprocessor.fit(X)

    assert "explicit_drop" in preprocessor.learned_drop_cols_, (
        "explicit_drop не попал в learned_drop_cols_."
    )
    assert "auto_const" in preprocessor.learned_drop_cols_, (
        "auto_const (константная) не попала в learned_drop_cols_."
    )
