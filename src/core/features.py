from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from omegaconf import DictConfig

from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


# ============================================================
# 1. ПРЕПРОЦЕССИНГ (Очистка, пропуски, выбросы)
# ============================================================

class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    Умный класс очистки. Автоматически различает числовые и категориальные признаки,
    удаляет мусорные колонки и обрезает выбросы.
    """

    def __init__(self, config: DictConfig):
        self.cfg = config.data.tabular
        self.num_strategy = getattr(self.cfg, 'num_fill_strategy', 'median')
        self.cat_strategy = getattr(self.cfg, 'cat_fill_strategy', 'unknown')
        self.drop_cols = list(self.cfg.drop_cols) if self.cfg.drop_cols else []
        self.skip_imputation = set(getattr(self.cfg, 'skip_imputation_cols', []))

    def fit(self, X: pd.DataFrame, y=None):

        self.fill_values_ = {}
        self.outlier_bounds_ = {}

        X_fit = X.drop(columns=self.drop_cols, errors='ignore')

        # ==========================================================
        # ШАГ 1: УДАЛЕНИЕ МУСОРНЫХ КОЛОНОК (Пустые и Константные)
        # ==========================================================

        # 1. Анализ пропусков и констант
        missing_frac = X_fit.isnull().mean()
        max_missing = getattr(self.cfg, 'max_missing_pct', 0.90)
        cols_to_drop_missing = missing_frac[missing_frac > max_missing].index.tolist()

        cols_to_drop_const = []
        max_const = getattr(self.cfg, 'max_constant_pct', 0.99)
        for col in X_fit.columns:
            # Защита от пустых колонок
            val_counts = X_fit[col].value_counts(normalize=True, dropna=False)
            if not val_counts.empty and val_counts.iloc[0] > max_const:
                cols_to_drop_const.append(col)

        # Собираем все "плохие" колонки
        self.learned_drop_cols_ = list(set(self.drop_cols + cols_to_drop_missing + cols_to_drop_const))

        # Дропаем их из X_fit, чтобы не считать по ним статистики
        X_fit = X.drop(columns=self.learned_drop_cols_, errors='ignore')

        # ==========================================================
        # Шаг 2: РАСЧЕТ ПРОПУСКОВ (Основная логика)
        # ==========================================================
        numeric_cols = set(X_fit.select_dtypes(include=[np.number]).columns)
        for col in X_fit.columns:
            if col in self.skip_imputation: continue
            if col in numeric_cols:
                val = X_fit[col].median() if self.num_strategy == 'median' else X_fit[col].mean()
                self.fill_values_[col] = 0 if pd.isna(val) else val
            else:
                self.fill_values_[col] = X_fit[col].mode()[0] if self.cat_strategy == 'mode' and not X_fit[
                    col].mode().empty else 'Unknown'

        # ==========================================================
        # ШАГ 3: РАСЧЕТ ГРАНИЦ ДЛЯ ВЫБРОСОВ (Только для чисел)
        # ==========================================================
        if getattr(self.cfg, 'outlier_method', 'none') == 'zscore':
            thresh = getattr(self.cfg, 'outlier_threshold', 3.0)
            for col in numeric_cols:
                if col in X_fit.columns:
                    mean, std = X_fit[col].mean(), X_fit[col].std()
                    self.outlier_bounds_[col] = (mean - thresh * std, mean + thresh * std)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Проверка, был ли вызван fit()
        check_is_fitted(self, ['fill_values_', 'learned_drop_cols_'])

        X_transformed = X.copy()

        # 1. Удаляем все плохие колонки (которые определили в fit)
        X_transformed = X_transformed.drop(columns=self.learned_drop_cols_, errors='ignore')

        # 2. Заполняем пропуски
        X_transformed = X_transformed.fillna(self.fill_values_)

        # ==========================================================
        # ШАГ 3 (ПРОДОЛЖЕНИЕ): ОБРЕЗКА ВЫБРОСОВ (Clipping)
        # ==========================================================
        if hasattr(self, 'outlier_bounds_') and self.outlier_bounds_:
            for col, (lower, upper) in self.outlier_bounds_.items():
                if col in X_transformed.columns:
                    X_transformed[col] = X_transformed[col].clip(lower=lower, upper=upper)

        return X_transformed

# ============================================================
# 2. ПРЕПРОЦЕССИНГ Кастомный препроцессинг
# ============================================================

class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Кастомный импьютер для колонок, пропущенных в TabularPreprocessor.
    Пример логики: заполнение пропусков средним значением с группировкой по другой колонке.
    """

    def __init__(self, target_cols: list[str], group_col: str):
        self.target_cols = target_cols  # Колонки с пропусками (например, 'salary', 'age')
        self.group_col = group_col  # Колонка для группировки (например, 'city')

        # Словарь для хранения вычисленных средних на Train-выборке
        self.group_means_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        logger.info(f"Обучение CustomImputer: группировка по {self.group_col}...")

        # Защита: проверяем, есть ли колонка группировки в данных
        if self.group_col not in X.columns:
            raise ValueError(f"Колонка {self.group_col} не найдена в датасете!")

        # Вычисляем средние значения только на X_fit (чтобы не было утечки данных)
        for col in self.target_cols:
            if col in X.columns:
                # Сохраняем словарь: {город1: средняя_зп, город2: средняя_зп}
                self.group_means_[col] = X.groupby(self.group_col)[col].mean().to_dict()

                # Запоминаем глобальное среднее на случай, если в Test попадется
                # совершенно новый город, которого не было в Train
                self.group_means_[f"{col}_global_mean"] = X[col].mean()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Применение CustomImputer...")
        X_transformed = X.copy()

        for col in self.target_cols:
            if col in X_transformed.columns:
                # 1. Заполняем пропуски словарем с группировкой
                mapped_values = X_transformed[self.group_col].map(self.group_means_[col])
                X_transformed[col] = X_transformed[col].fillna(mapped_values)

                # 2. Если остались пропуски (попался новый город из Test),
                # бьем глобальным средним значением
                global_mean = self.group_means_.get(f"{col}_global_mean", 0)
                X_transformed[col] = X_transformed[col].fillna(global_mean)

        return X_transformed

# ============================================================
# 3. ИНЖЕНЕРИЯ ПРИЗНАКОВ (Генерация новых фичей)
# ============================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Генерация новых бизнес-фичей.
    Обычно не требует метода fit, так как трансформации построчные.
    """

    def __init__(self, config: DictConfig):
        self.cfg = config

    def fit(self, X: pd.DataFrame, y=None):
        # Нам не нужно собирать статистику для генерации фичей
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Генерация новых признаков...")
        X_transformed = X.copy()

        # --- ПРИМЕРЫ (Плейсхолдеры для шаблона) ---

        # Если есть колонка с датой, вытаскиваем месяц/день недели
        # if 'created_at' in X_transformed.columns:
        #     X_transformed['created_at'] = pd.to_datetime(X_transformed['created_at'])
        #     X_transformed['month'] = X_transformed['created_at'].dt.month
        #     X_transformed['is_weekend'] = X_transformed['created_at'].dt.dayofweek >= 5

        # Пример математической фичи
        # if 'income' in X_transformed.columns and 'expenses' in X_transformed.columns:
        #     X_transformed['savings_ratio'] = X_transformed['income'] / (X_transformed['expenses'] + 1e-5)

        return X_transformed