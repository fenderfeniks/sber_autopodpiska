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

        # МАСТЕР-ПЕРЕКЛЮЧАТЕЛЬ: защита от случайного сэмплирования в проде
        self.is_dev_mode = (self.cfg.env == "dev")

        # Если мы в dev, берем процент из конфига, иначе жестко 1.0 (100%)
        self.sample_pct = self.data_cfg.sample_pct if self.is_dev_mode else 1.0

    def _get_db_engine(self):
        """Создает подключение к БД на основе защищенного конфига."""
        if self.cfg.database is None:
            raise ValueError("database не настроен в конфиге, но запрошено подключение к БД.")

        db = self.cfg.database
        conn_string = f"postgresql://{db.user}:{quote_plus(db.password)}@{db.host}:{db.port}/{db.name}"
        return create_engine(conn_string)

    def load_data(self) -> pd.DataFrame:
        """
        Умный и ленивый загрузчик.
        Сэмплирует данные до загрузки в RAM и автоматически удаляет мусорные строки.
        """
        df = None  # Инициализируем пустой датафрейм

        # ============================================================
        # 1. ЗАГРУЗКА И СЭМПЛИРОВАНИЕ (Lazy Loading)
        # ============================================================
        if self.cfg.paths.table_name:
            table_name = self.cfg.paths.table_name
            logger.info(f"Загрузка из БД, таблица: {table_name}")

            #Валидация имени таблицы (Защита от SQL Injection)
            if not re.fullmatch(r"[a-zA-Z0-9_]+", table_name):
                raise ValueError(f"Недопустимое имя таблицы базы данных: {table_name}")

            engine = self._get_db_engine()

            if self.sample_pct < 1.0:
                logger.info(f"[DEV MODE] SQL-сэмплирование: {self.sample_pct * 100}%")
                # === РАСКОММЕНТИРУЙ НУЖНЫЙ ДИАЛЕКТ ===

                # 1. PostgreSQL / SQLite
                query = f"SELECT * FROM {table_name} WHERE random() < {self.sample_pct}"

                # 2. MySQL / SQL Server
                # query = f"SELECT * FROM {table_name} ORDER BY RAND() LIMIT {int(total_rows * self.sample_pct)}"

                # 3. Oracle
                # query = f"SELECT * FROM {table_name} SAMPLE({self.sample_pct * 100})"
            else:
                query = f"SELECT * FROM {table_name}"

            df = pd.read_sql(query, engine)

        else:
            file_path = PROJECT_ROOT / self.cfg.paths.raw_dir / self.cfg.paths.data_file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Файл не найден: {file_path}")

            logger.info(f"Чтение файла: {file_path.name}")

            # --- CSV ---
            if file_path.suffix == '.csv':
                if self.sample_pct < 1.0:
                    logger.info(f"[DEV MODE] Быстрое сэмплирование CSV (DuckDB): {self.sample_pct * 100}%")
                    # Устранили горлышко с lambda! DuckDB читает CSV в десятки раз быстрее.
                    query = f"SELECT * FROM read_csv_auto('{file_path}') USING SAMPLE {self.sample_pct * 100} PERCENT (bernoulli)"
                    df = duckdb.query(query).to_df()
                else:
                    # В проде читаем весь файл через оптимизированный движок pyarrow (если установлен),
                    # иначе Pandas сам откатится на стандартный C-engine.
                    try:
                        df = pd.read_csv(file_path, engine='pyarrow')
                    except ImportError:
                        logger.warning("pyarrow не установлен. Используется стандартный engine Pandas.")
                        df = pd.read_csv(file_path)

            # --- PARQUET ---
            elif file_path.suffix in ['.parquet', '.pqt']:
                if self.sample_pct < 1.0:
                    logger.info(f"[DEV MODE] Ленивое чтение Parquet (DuckDB): {self.sample_pct * 100}%")
                    query = f"SELECT * FROM '{file_path}' USING SAMPLE {self.sample_pct * 100} PERCENT (bernoulli)"
                    df = duckdb.query(query).to_df()
                else:
                    df = pd.read_parquet(file_path)

            else:
                raise ValueError(f"Формат {file_path.suffix} не поддерживается.")

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