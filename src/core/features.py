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
    Класс технической предобработки. 
    Отвечает за умную импутацию пропусков по бизнес-правилам, 
    схлопывание редких категорий и фильтрацию констант.
    """

    def __init__(self, config: DictConfig):
        self.full_cfg = config
        self.cfg = config.data.tabular
        
        self.num_strategy = getattr(self.cfg, 'num_fill_strategy', 'median')
        self.cat_strategy = getattr(self.cfg, 'cat_fill_strategy', 'unknown')
        
        self.drop_cols = list(self.cfg.drop_cols) if self.cfg.drop_cols else []
        self.skip_imputation = set(getattr(self.cfg, 'skip_imputation_cols', []))
        self.top_n_categories = getattr(self.cfg, 'top_n_categories', 12)

    def fit(self, X: pd.DataFrame, y=None):
        self.fill_values_ = {}
        self.outlier_bounds_ = {}
        self.top_categories_map_ = {}

        # Базовые копии для анализа статистик трейна
        X_fit = X.copy()
        
        # 1. Расчет статистик для умной импутации площади экрана по брендам
        self.global_screen_area_median_ = X_fit['screen_area'].median() if 'screen_area' in X_fit.columns else 0
        if 'device_brand' in X_fit.columns and 'screen_area' in X_fit.columns:
            # Запоминаем медианы экранов для каждого бренда на трейне
            self.brand_screen_medians_ = X_fit.groupby('device_brand')['screen_area'].median().to_dict()
        else:
            self.brand_screen_medians_ = {}

        # Исключаем технические колонки из анализа пропусков и констант
        technical_cols = self.drop_cols + ['session_id', 'client_id', 'visit_date', 'visit_time', 'device_screen_resolution']
        X_fit = X_fit.drop(columns=technical_cols, errors='ignore')

        # ==========================================================
        # ШАГ 1: АНАЛИЗ МУСОРНЫХ КОЛОНОК (Пропуски и Константы)
        # ==========================================================
        missing_frac = X_fit.isnull().mean()
        max_missing = getattr(self.cfg, 'max_missing_pct', 0.90)
        cols_to_drop_missing = missing_frac[missing_frac > max_missing].index.tolist()

        cols_to_drop_const = []
        max_const = getattr(self.cfg, 'max_constant_pct', 0.99)
        for col in X_fit.columns:
            val_counts = X_fit[col].value_counts(normalize=True, dropna=False)
            if not val_counts.empty and val_counts.iloc[0] > max_const:
                cols_to_drop_const.append(col)

        # Формируем финальный черный список колонок
        self.learned_drop_cols_ = list(set(technical_cols + cols_to_drop_missing + cols_to_drop_const))
        X_fit = X_fit.drop(columns=self.learned_drop_cols_, errors='ignore')

        # ==========================================================
        # ШАГ 2: КАРТИРОВАНИЕ ТОП-КАТЕГОРИЙ (Схлопывание хвостов)
        # ==========================================================
        categorical_cols = X_fit.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            top_vals = X_fit[col].value_counts().index[:self.top_n_categories].tolist()
            self.top_categories_map_[col] = top_vals

        # ==========================================================
        # Шаг 3: РАСЧЕТ ДЕФОЛТНЫХ ЗАГЛУШЕК ДЛЯ ОСТАВШИХСЯ НАХОДОК
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
        # ШАГ 4: РАСЧЕТ ГРАНИЦ ДЛЯ ВЫБРОСОВ
        # ==========================================================
        if getattr(self.cfg, 'outlier_method', 'none') == 'zscore':
            thresh = getattr(self.cfg, 'outlier_threshold', 3.0)
            for col in numeric_cols:
                if col in X_fit.columns:
                    mean, std = X_fit[col].mean(), X_fit[col].std()
                    if std > 0:
                        self.outlier_bounds_[col] = (mean - thresh * std, mean + thresh * std)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ['fill_values_', 'learned_drop_cols_', 'top_categories_map_'])
        X_transformed = X.copy()

        # ==========================================================
        # УМНАЯ ИМПУТАЦИЯ ПО СЛОЖНЫМ ПРАВИЛАМ БИЗНЕС-ЛОГИКИ
        # ==========================================================
        
        # 1. Первичная подстраховка для брендов устройств
        if 'device_brand' in X_transformed.columns:
            X_transformed['device_brand'] = X_transformed['device_brand'].fillna('Unknown')

        # 2. Заполнение площади экрана по медиане конкретного БРЕНДА
        if 'screen_area' in X_transformed.columns:
            brand_medians = X_transformed['device_brand'].map(self.brand_screen_medians_)
            brand_medians = brand_medians.fillna(self.global_screen_area_median_)
            X_transformed['screen_area'] = X_transformed['screen_area'].fillna(brand_medians)

        # 3. Направленное заполнение device_os на основе категорий и брендов
        if 'device_os' in X_transformed.columns:
            is_desktop = X_transformed['device_category'] == 'desktop'
            is_mobile = X_transformed['device_category'].isin(['mobile', 'tablet'])
            is_apple = X_transformed['device_brand'].str.lower() == 'apple'

            X_transformed['device_os'] = np.where(is_desktop & X_transformed['device_os'].isnull(), 'Windows', X_transformed['device_os'])
            X_transformed['device_os'] = np.where(is_mobile & is_apple & X_transformed['device_os'].isnull(), 'iOS', X_transformed['device_os'])
            X_transformed['device_os'] = np.where(is_mobile & ~is_apple & X_transformed['device_os'].isnull(), 'Android', X_transformed['device_os'])

        # ==========================================================
        # ТЕХНИЧЕСКАЯ ОЧИСТКА И ФИЛЬТРАЦИЯ
        # ==========================================================
        
        # 4. Применяем зафиксированное на трейне схлопывание редких категорий
        for col, top_vals in self.top_categories_map_.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].where(X_transformed[col].isin(top_vals), 'other_collapsed')

        # 5. Дропаем все ненужные, пустые и константные колонки разом
        X_transformed = X_transformed.drop(columns=self.learned_drop_cols_, errors='ignore')

        # 6. Финальное заполнение базовых NaN (если они где-то остались)
        X_transformed = X_transformed.fillna(self.fill_values_)

        # 7. Обрезка экстремальных выбросов (Clipping)
        if hasattr(self, 'outlier_bounds_') and self.outlier_bounds_:
            for col, (lower, upper) in self.outlier_bounds_.items():
                if col in X_transformed.columns:
                    X_transformed[col] = X_transformed[col].clip(lower=lower, upper=upper)

        return X_transformed


# ============================================================
# 2. ИНЖЕНЕРИЯ ПРИЗНАКОВ (Генерация новых фичей)
# ============================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Генерация бизнес-признаков, парсинг гео, расчет площади экрана.
    Данные по городам динамически подгружаются из Hydra-конфига и обрабатываются в transform.
    """
    def __init__(self, config: DictConfig):
        self.cfg = config
        
        # 1. Безопасно достаем конфигурацию по гео и девайсам через .get()
        tabular_cfg = config.get('data', {}).get('tabular', {})
        geo_cfg = tabular_cfg.get('geo', {})
        devices_cfg = tabular_cfg.get('devices', {})
        
        self.cis_countries = list(geo_cfg.get('cis', []))
        self.mobile_cats = list(devices_cfg.get('mobile_categories', ['mobile', 'tablet']))

        # 2. Извлекаем справочник городов прямо из пути cfg.data.tabular
        # Конвертируем в нативный dict для максимальной скорости работы .map() в Pandas
        city_markets_cfg = tabular_cfg.get('city_markets', {})
        self.city_markets = OmegaConf.to_container(city_markets_cfg, resolve=True) if city_markets_cfg else {}
        
        # Загружаем дефолтный профиль на случай редких/пропущенных локаций
        defaults_cfg = tabular_cfg.get('defaults_fallback', {})
        self.city_defaults = OmegaConf.to_container(defaults_cfg, resolve=True) if defaults_cfg else {
            'has_metro': 0, 'population_2021': 100000, 'avg_salary_2021': 33000, 'cars_per_family': 0.65
        }

    def fit(self, X: pd.DataFrame, y=None):
        # Нам не нужно собирать статистику, так как трансформации детерминированные
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Запуск Feature Engineering пайплайна...")
        X_transformed = X.copy()

        # ==========================================================
        # 1. СИНХРОНИЗАЦИЯ ЗАГЛУШЕК (Убираем конфликт Other)
        # ==========================================================
        for col in X_transformed.select_dtypes(include=['object']).columns:
            X_transformed[col] = X_transformed[col].replace(['(Other)', '(not set)', 'other'], np.nan)

        # ==========================================================
        # 2. ПАРСИНГ ПЛОЩАДИ ЭКРАНА
        # ==========================================================
        if 'device_screen_resolution' in X_transformed.columns:
            res_split = X_transformed['device_screen_resolution'].astype(str).str.split('x', expand=True)
            if res_split.shape[1] >= 2:
                width = pd.to_numeric(res_split[0], errors='coerce')
                height = pd.to_numeric(res_split[1], errors='coerce')
                X_transformed['screen_area'] = width * height
            else:
                X_transformed['screen_area'] = np.nan
        else:
            X_transformed['screen_area'] = np.nan

        # ==========================================================
        # 3. ГЕО-ЗОНЫ И МОБИЛЬНОСТЬ
        # ==========================================================
        if 'geo_country' in X_transformed.columns:
            conditions = [
                X_transformed['geo_country'] == 'Russia',
                X_transformed['geo_country'].isin(self.cis_countries)
            ]
            choices = ['russia', 'cis']
            X_transformed['geo_zone'] = np.select(conditions, choices, default='other_countries')
        else:
            X_transformed['geo_zone'] = 'other_countries'

        if 'device_category' in X_transformed.columns:
            X_transformed['is_mobile_device'] = X_transformed['device_category'].isin(self.mobile_cats).astype(int)
        else:
            X_transformed['is_mobile_device'] = 0

        # ==========================================================
        # 4. КАЛЕНДАРНЫЕ И ВРЕМЕННЫЕ Признаки
        # ==========================================================
        if 'visit_date' in X_transformed.columns:
            date_series = pd.to_datetime(X_transformed['visit_date'], errors='coerce')
            X_transformed['day_of_week'] = date_series.dt.dayofweek.fillna(0).astype(int)
            X_transformed['is_weekend'] = X_transformed['day_of_week'].isin([5, 6]).astype(int)

        if 'visit_time' in X_transformed.columns:
            hours = pd.to_datetime(X_transformed['visit_time'], format='%H:%M:%S', errors='coerce').dt.hour
            if hours.isnull().all():
                hours = pd.to_numeric(X_transformed['visit_time'], errors='coerce').fillna(12)
            X_transformed['is_night'] = hours.isin([23, 0, 1, 2, 3, 4, 5]).astype(int)

        # ==========================================================
        # 5. МАТЕМАТИКА СКОРОСТИ КЛИКОВ И КРОССЫ
        # ==========================================================
        if 'last_hit_time_ms' in X_transformed.columns and 'first_hit_time_ms' in X_transformed.columns:
            X_transformed['session_duration_ms'] = X_transformed['last_hit_time_ms'] - X_transformed['first_hit_time_ms']
            if 'total_hits_count' in X_transformed.columns:
                X_transformed['ms_per_hit'] = np.where(
                    X_transformed['total_hits_count'] > 0,
                    X_transformed['session_duration_ms'] / X_transformed['total_hits_count'],
                    0
                )

        if 'device_category' in X_transformed.columns and 'device_brand' in X_transformed.columns:
            X_transformed['dev_category_brand'] = (
                X_transformed['device_category'].astype(str) + "_" + X_transformed['device_brand'].astype(str)
            )

        # ==========================================================
        # 6. ВСТРОЕННОЕ ОБОГАЩЕНИЕ ДЕМОГРАФИЕЙ ГОРОДОВ (Из конфига)
        # ==========================================================
        if 'geo_city' in X_transformed.columns:
            # Нормализуем строку города для точного мэтчинга с ключами YAML
            city_normalized = X_transformed['geo_city'].astype(str).str.lower().str.strip()
            
            # Маппим признаки, используя кэшированный нативный python-dict
            X_transformed['has_metro'] = city_normalized.map(
                lambda x: self.city_markets.get(x, {}).get('has_metro', self.city_defaults['has_metro'])
            )
            X_transformed['city_population'] = city_normalized.map(
                lambda x: self.city_markets.get(x, {}).get('population_2021', self.city_defaults['population_2021'])
            )
            X_transformed['city_avg_salary'] = city_normalized.map(
                lambda x: self.city_markets.get(x, {}).get('avg_salary_2021', self.city_defaults['avg_salary_2021'])
            )
            X_transformed['city_cars_per_family'] = city_normalized.map(
                lambda x: self.city_markets.get(x, {}).get('cars_per_family', self.city_defaults['cars_per_family'])
            )
        else:
            # Если колонки нет, заполняем дефолтными значениями
            X_transformed['has_metro'] = self.city_defaults['has_metro']
            X_transformed['city_population'] = self.city_defaults['population_2021']
            X_transformed['city_avg_salary'] = self.city_defaults['avg_salary_2021']
            X_transformed['city_cars_per_family'] = self.city_defaults['cars_per_family']

        # 7. Фича-пропорция (интерес юзера к авто относительно среднего по городу)
        if 'total_car_views' in X_transformed.columns:
            X_transformed['user_vs_city_car_interest'] = X_transformed['total_car_views'] / (X_transformed['city_cars_per_family'] + 1e-5)

        return X_transformed