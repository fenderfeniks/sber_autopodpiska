from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import psycopg2
from pathlib import Path
import pandas as pd
from src.config import load_config

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"
cfg = load_config(CONFIG_PATH)


def get_engine() -> Engine:
    return create_engine(
        f"postgresql://{cfg.database.user}:{cfg.database.password}"
        f"@{cfg.database.host}:{cfg.database.port}/{cfg.database.name}"
    )


def get_connection():
    return psycopg2.connect(
        host=cfg.database.host,
        port=cfg.database.port,
        dbname=cfg.database.name,
        user=cfg.database.user,
        password=cfg.database.password
    )


def load_df_to_db(df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"{table_name} загружена")


def read_table(table_name: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


def execute_query(query: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)