from __future__ import annotations

import logging
from abc import ABC, abstractmethod
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """
    Базовый класс для источников данных. Определяет контракт загрузки
    и общую пост-обработку (smart row dropping по доле пропусков).
    """

    def __init__(self, cfg: DictConfig, project_root):
        self.cfg = cfg
        self.PROJECT_ROOT = project_root

    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...

    def _apply_missing_row_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMART ROW DROPPING: удаление строк с избыточной долей пропусков по столбцам."""
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
        return df


class SqlAggregatedDataSource(BaseDataSource):
    """
    Загрузка агрегированной витрины из PostgreSQL (join таблиц + бизнес-логика
    формирования таргета), с локальным кэшированием в parquet.
    """

    def __init__(self, cfg: DictConfig, project_root):
        super().__init__(cfg, project_root)
        self.seed = cfg.seed
        self.aggrigation_version = getattr(cfg.data.tabular, 'aggrigation_version', 'v1.0.0')
        self.sample_pct = cfg.data.sample_pct

    def _get_db_engine(self):
        from sqlalchemy import create_engine
        from urllib.parse import quote_plus
        if self.cfg.database is None:
            raise ValueError("database не настроен в конфиге, но запрошено подключение к БД.")

        db = self.cfg.database
        conn_string = f"postgresql://{db.user}:{quote_plus(db.password)}@{db.host}:{db.port}/{db.name}"
        return create_engine(conn_string)

    def load(self, sql_file_name: str = "features/dirty_baseline_aggregation.sql") -> pd.DataFrame:
        cache_dir = self.PROJECT_ROOT / self.cfg.paths.processed_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file_name = f"aggregated_features_{self.aggrigation_version}_sample_{self.sample_pct}.parquet"
        cache_path = cache_dir / cache_file_name

        if cache_path.exists():
            logger.info(f"[CACHE] Найден локальный кэш витрины {self.aggrigation_version}. Загрузка...")
            df = pd.read_parquet(cache_path)
            logger.info(f"[CACHE] Успешно загружено из кэша. Размерность: {df.shape}")
            return self._apply_missing_row_filter(df)

        logger.info(f"[DB_MODE] Кэш не найден. Запуск агрегации в PostgreSQL "
                    f"(Версия фичей: {self.aggrigation_version})...")

        engine = self._get_db_engine()

        if getattr(self.cfg.paths, "features_query_path", None):
            sql_file_name = self.cfg.paths.features_query_path

        sql_path = self.PROJECT_ROOT / self.cfg.paths.sql_dir / sql_file_name
        logger.info(f"Чтение SQL-скрипта: {sql_path.name}")

        if not sql_path.exists():
            raise FileNotFoundError(f"SQL файл не найден: {sql_path}")

        with open(sql_path, "r", encoding="utf-8") as f:
            raw_sql_content = f.read()

        target_actions = self.cfg.data.tabular.target_event_actions
        if not target_actions:
            raise ValueError("В конфиге configs.data.tabular.target_event_actions не указаны целевые действия!")

        formatted_actions = ", ".join([f"'{action}'" for action in target_actions])
        target_actions_condition = f"event_action IN ({formatted_actions})"

        base_query = raw_sql_content.format(
            raw_hits_table=self.cfg.paths.raw_hits_table,
            raw_sessions_table=self.cfg.paths.raw_sessions_table,
            target_actions_condition=target_actions_condition
        )

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

        logger.info("Выполнение агрегации данных внутри PostgreSQL...")
        df = pd.read_sql(query, engine)
        logger.info(f"Данные загружены из БД. Размерность: {df.shape}")

        logger.info(f"[CACHE] Сохранение агрегированных данных в локальный кэш: {cache_path}")
        df.to_parquet(cache_path, index=False)

        return self._apply_missing_row_filter(df)


class FlatFileDataSource(BaseDataSource):
    """
    Загрузка данных из плоского файла (CSV/Parquet), без join и агрегации.
    Формат определяется по расширению cfg.paths.data_file_name.
    """

    def __init__(self, cfg: DictConfig, project_root):
        super().__init__(cfg, project_root)
        self.sample_pct = cfg.data.sample_pct

    def load(self) -> pd.DataFrame:
        import duckdb

        file_path = self.PROJECT_ROOT / self.cfg.paths.raw_dir / self.cfg.paths.data_file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Файл данных не найден по пути: {file_path}")

        logger.info(f"[FILE_MODE] Чтение файла: {file_path.name}")

        if file_path.suffix == '.csv':
            if self.sample_pct < 1.0:
                logger.info(f"[DEV MODE] Быстрое сэмплирование CSV через DuckDB: {self.sample_pct * 100}%")
                query = (f"SELECT * FROM read_csv_auto('{file_path.as_posix()}') "
                         f"USING SAMPLE {self.sample_pct * 100} PERCENT (bernoulli)")
                df = duckdb.query(query).to_df()
            else:
                try:
                    df = pd.read_csv(file_path, engine='pyarrow')
                except ImportError:
                    df = pd.read_csv(file_path)

        elif file_path.suffix in ['.parquet', '.pqt']:
            if self.sample_pct < 1.0:
                logger.info(f"[DEV MODE] Ленивое чтение Parquet через DuckDB: {self.sample_pct * 100}%")
                query = (f"SELECT * FROM '{file_path.as_posix()}' "
                         f"USING SAMPLE {self.sample_pct * 100} PERCENT (bernoulli)")
                df = duckdb.query(query).to_df()
            else:
                df = pd.read_parquet(file_path)

        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}")

        return self._apply_missing_row_filter(df)