from __future__ import annotations

import logging
from urllib.parse import quote_plus
from typing import Tuple
import re

import pandas as pd
from sqlalchemy import create_engine
import duckdb
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

from core.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)


class UniversalDataLoader:
    """
    Класс для загрузки сырых данных из CSV, Parquet или баз данных,
    а также для их первичного разбиения на выборки.
    """

    def __init__(self, config: DictConfig):
        self.cfg = config
        self.seed = config.seed

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

    def load_data(self, sql_file_name: str = None, query_params: dict = None) -> pd.DataFrame:
        """
        Умный и ленивый загрузчик.
        Сэмплирует данные до загрузки в RAM и автоматически удаляет мусорные строки.
        """
        df = None  # Инициализируем пустой датафрейм

        # ============================================================
        # 1. ЗАГРУЗКА И СЭМПЛИРОВАНИЕ (Lazy Loading)
        # ============================================================
        engine = self._get_db_engine()
        
        # 1. Читаем базовый текст SQL-запроса из файла, если он передан
        if sql_file_name:
            sql_path = PROJECT_ROOT / self.cfg.paths.sql_dir / sql_file_name
            logger.info(f"Загрузка данных с помощью SQL-скрипта: {sql_path.name}")
            with open(sql_path, "r", encoding="utf-8") as f:
                base_query = f.read()
        else:
            # Если файл не передан, берем дефолтную таблицу из конфига путей
            table_name = self.cfg.paths.table_name
            if not table_name:
                raise ValueError("В конфигурации не указаны ни table_name, ни sql_file_name")
            if not re.fullmatch(r"[a-zA-Z0-9_]+", table_name):
                raise ValueError(f"Недопустимое имя таблицы: {table_name}")
            base_query = f"SELECT * FROM {table_name}"

        # 2. Применяем смарт-сэмплирование по пользователям
        if self.sample_pct < 1.0:
            logger.info(f"[DEV MODE] Сэмплирование по пользователям: {self.sample_pct * 100}%")
            query = f"""
            WITH full_dataset AS (
                {base_query}
            ),
            sampled_users AS (
                SELECT client_id 
                FROM ga_sessions 
                GROUP BY client_id 
                HAVING random() < {self.sample_pct}
            )
            SELECT f.* FROM full_dataset f
            JOIN sampled_users s ON f.client_id = s.client_id;
            """
        else:
            query = base_query

        # 3. Выполняем результирующий запрос в СУБД
        df = pd.read_sql(query, engine, params=query_params)

        # ============================================================
        # 2. SMART ROW DROPPING (Удаление слишком пустых строк)
        # ============================================================
        # Защита: проверяем, что данные табличные (не NLP) и параметр существует
        if self.cfg.data.tabular and hasattr(self.cfg.data.tabular, 'max_row_missing_pct'):
            row_missing_threshold = self.cfg.data.tabular.max_row_missing_pct

            if row_missing_threshold < 1.0:
                initial_rows = len(df)
                total_cols = df.shape[1]

                # Считаем минимально допустимое количество ЗАПОЛНЕННЫХ ячеек
                # Например: из 10 колонок при пороге 0.5 (50%) хотя бы 5 должны быть не NaN
                min_non_nulls = int(total_cols * (1.0 - row_missing_threshold))

                # Векторизованное удаление на C-уровне (thresh требует указать кол-во НЕ-пустых)
                df = df.dropna(thresh=min_non_nulls).reset_index(drop=True)

                dropped_rows = initial_rows - len(df)
                if dropped_rows > 0:
                    logger.info(
                        f"[SMART DROP] Удалено {dropped_rows} мусорных строк "
                        f"(содержали > {row_missing_threshold * 100}% пропусков)."
                    )

        return df

    def get_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разбивает датафрейм на Train, Val и Test с явным контролем стратификации
        на основе типа задачи из конфига.
        """
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