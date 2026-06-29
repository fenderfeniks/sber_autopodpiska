from __future__ import annotations

import logging
from urllib.parse import quote_plus
from typing import Tuple

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class UniversalDataLoader:
    """
    Класс для загрузки сырых данных из CSV, Parquet или баз данных,
    а также для их первичного разбиения на выборки.
    """

    def __init__(self, config: DictConfig, project_root):
        self.PROJECT_ROOT = project_root
        self.cfg = config
        self.seed = config.seed
        self.aggrigation_version = getattr(config.data.tabular, 'aggrigation_version', 'v1.0.0')
        self.data_cfg = config.data

        # Если мы в dev, берем процент из конфига, иначе жестко 1.0 (100%)
        self.sample_pct = self.data_cfg.sample_pct

    def _get_db_engine(self):
        """Создает подключение к БД на основе защищенного конфига."""
        if self.cfg.database is None:
            raise ValueError("database не настроен в конфиге, но запрошено подключение к БД.")

        db = self.cfg.database
        conn_string = f"postgresql://{db.user}:{quote_plus(db.password)}@{db.host}:{db.port}/{db.name}"
        return create_engine(conn_string)

    def filter_bad_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление строк, у которых процент пропусков превышает допустимый порог."""
        max_row_missing = getattr(self.cfg.data.tabular, 'max_row_missing_pct', 0.50)
        
        # Считаем долю пропусков в каждой строке (ось axis=1)
        row_missing_frac = df.isnull().mean(axis=1)
        
        # Маска для выживших строк
        good_rows_mask = row_missing_frac <= max_row_missing
        
        removed_count = len(df) - good_rows_mask.sum()
        if removed_count > 0:
            logger.info(
                f"[DATA CLEANING] Удалено {removed_count} строк, содержащих > {max_row_missing*100}% пропусков."
            )
            df = df[good_rows_mask].reset_index(drop=True)
            
        return df

    def load_data(self, sql_file_name: str = "features/dirty_baseline_aggregation.sql", query_params: dict = None) -> pd.DataFrame:
        """
        Умный загрузчик. Читает SQL-файл, динамически подставляет таргет-действия 
        из Hydra-конфига, выполняет агрегацию в СУБД и возвращает DataFrame.
        """
        # 1. Формируем путь к кэшу
        cache_dir = self.PROJECT_ROOT / self.cfg.paths.processed_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Имя файла включает версию и флаг сэмпла (чтобы дев-версия не затерла полный датасет)
        cache_file_name = f"aggregated_features_{self.aggrigation_version}_sample_{self.sample_pct}.parquet"
        cache_path = cache_dir / cache_file_name

        # 2. Проверяем, существует ли уже этот кэш
        if cache_path.exists():
            logger.info(f"[CACHE] Найден локальный кэш витрины {self.aggrigation_version}. Загрузка из файла...")
            df = pd.read_parquet(cache_path)
            logger.info(f"[CACHE] Данные успешно загружены из кэша. Размерность: {df.shape}")
            return df
        
        logger.info(f"[DB] Кэш не найден. Запуск агрегации внутри PostgreSQL (Версия фичей: {self.aggrigation_version})...")

        engine = self._get_db_engine()

        if self.cfg.paths.features_query_path:
            sql_file_name = self.cfg.paths.features_query_path

        # 1. Читаем SQL-запрос из файла
        sql_path = self.PROJECT_ROOT / self.cfg.paths.sql_dir / sql_file_name
        logger.info(f"Чтение SQL-скрипта агрегации: {sql_path.name}")
        
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL файл не найден по пути: {sql_path}")
            
        with open(sql_path, "r", encoding="utf-8") as f:
            raw_sql_content = f.read()

        # 2. Формируем динамическое SQL-условие для таргета на основе списка из конфига
        # Достаем список из configs.data.tabular.target_event_actions
        target_actions = self.cfg.data.tabular.target_event_actions
        
        if not target_actions:
            raise ValueError("В конфиге configs.data.tabular.target_event_actions не указаны целевые действия!")
            
        # Превращаем список ['action1', 'action2'] в строку для SQL: event_action IN ('action1', 'action2')
        formatted_actions = ", ".join([f"'{action}'" for action in target_actions])
        target_actions_condition = f"event_action IN ({formatted_actions})"
        
        logger.info(f"Сформировано условие таргета: {target_actions_condition}")
            
        # Форматируем шаблон запроса
        base_query = raw_sql_content.format(
            raw_hits_table=self.cfg.paths.raw_hits_table,
            raw_sessions_table=self.cfg.paths.raw_sessions_table,
            target_actions_condition=target_actions_condition  # подставляем наше условие
        )

        # 3. Применяем смарт-сэмплирование по пользователям в DEV режиме
        if self.sample_pct < 1.0:
            logger.info(f"[DEV MODE] Сэмплирование по пользователям: {self.sample_pct * 100}%")
            query = f"""
            WITH base_dataset AS (
                {base_query}
            ),
            sampled_users AS (
                SELECT client_id 
                FROM {self.cfg.paths.raw_sessions_table} 
                GROUP BY client_id 
                HAVING random() < {self.sample_pct}
            )
            SELECT b.* FROM base_dataset b
            JOIN sampled_users s ON b.client_id = s.client_id;
            """
        else:
            query = base_query

        # 4. Выполняем результирующий запрос в СУБД
        logger.info("Выполнение агрегации данных внутри PostgreSQL...")
        df = pd.read_sql(query, engine)
        logger.info(f"Данные успешно загружены в RAM. Размерность: {df.shape}")

        # ============================================================
        # SMART ROW DROPPING (Удаление пустых строк)
        # ============================================================
        if self.cfg.data.tabular and hasattr(self.cfg.data.tabular, 'max_row_missing_pct'):
            row_missing_threshold = self.cfg.data.tabular.max_row_missing_pct

            if row_missing_threshold < 1.0:
                initial_rows = len(df)
                total_cols = df.shape[1]
                min_non_nulls = int(total_cols * (1.0 - row_missing_threshold))

                df = df.dropna(thresh=min_non_nulls).reset_index(drop=True)

                dropped_rows = initial_rows - len(df)
                if dropped_rows > 0:
                    logger.info(
                        f"[SMART DROP] Удалено {dropped_rows} строк "
                        f"(содержали > {row_missing_threshold * 100}% пропусков)."
                    )
        
        logger.info(f"[CACHE] Сохранение агрегированных данных в локальный кэш: {cache_path}")
        df.to_parquet(cache_path, index=False)

        return df

    def get_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разбивает датафрейм на Train, Val и Test с явным контролем стратификации
        на основе типа задачи из конфига.
        """
        df = self.filter_bad_rows(df)

        target_col = self.cfg.data.tabular.target_col if self.cfg.data.tabular else None
        task_type = self.cfg.task_type

        test_size = self.data_cfg.test_size
        val_size = self.data_cfg.val_size

        if test_size + val_size >= 1.0:
            raise ValueError(f"Сумма test_size и val_size должна быть < 1.0!")

        logger.info(f"Разбиение данных (Задача: {task_type})...")

        # --- ЯВНЫЙ КОНТРАКТ СТРАТИФИКАЦИИ ---
        stratify_col = None
        if target_col and target_col in df.columns:
            # Стратифицируем ТОЛЬКО если это явно классификация
            if task_type in ["binary", "multiclass", "sequence_classification"]:
                stratify_col = df[target_col]
            else:
                logger.info(f"Для задачи '{task_type}' стратификация отключена.")

        # --- БЕЗОПАСНЫЙ TEST SPLIT ---
        if test_size > 0:
            try:
                train_val_df, test_df = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=self.seed,
                    stratify=stratify_col
                )
            except ValueError as e:
                logger.warning(f"Ошибка стратификации на Test (вероятно, уникальный класс): {e}. Случайный сплит.")
                train_val_df, test_df = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=self.seed,
                    stratify=None
                )
        else:
            train_val_df = df
            test_df = pd.DataFrame(columns=df.columns)

        # --- БЕЗОПАСНЫЙ VAL SPLIT ---
        if val_size > 0:
            val_fraction = val_size / (1.0 - test_size)
            stratify_val = train_val_df[target_col] if stratify_col is not None else None

            try:
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=val_fraction,
                    random_state=self.seed,
                    stratify=stratify_val
                )
            except ValueError as e:
                logger.warning(f"Ошибка стратификации на Val: {e}. Выполняем случайный сплит.")
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=val_fraction,
                    random_state=self.seed,
                    stratify=None
                )
        else:
            train_df = train_val_df
            val_df = pd.DataFrame(columns=df.columns)

        # СБРОС ИНДЕКСОВ
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        logger.info(f"Размеры выборок: Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        return train_df, val_df, test_df