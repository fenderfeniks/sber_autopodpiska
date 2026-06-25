from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ============================================================
# БАЗОВЫЕ КОМПОНЕНТЫ (Вложенные схемы)
# ============================================================

@dataclass
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str


@dataclass
class DvcConfig:
    remote: str
    s3_bucket: Optional[str] = None
    endpoint_url: Optional[str] = None


@dataclass
class PathsConfig:
    # Базовые
    data_dir: str
    models_dir: str
    reports_dir: str
    logs_dir: str
    src_dir: str
    sql_dir: str

    # Поддиректории
    raw_dir: str
    processed_dir: str
    features_dir: str

    # Файлы
    data_file_name: str
    table_name: Optional[str] = None


# --- Блоки Данных ---

@dataclass
class TabularDataConfig:
    target_col: str
    drop_cols: List[str]

    num_fill_strategy: str
    cat_fill_strategy: str

    outlier_method: str
    outlier_threshold: float

    max_missing_pct: float = 0.90
    max_constant_pct: float = 0.99
    max_row_missing_pct: float = 0.50
    skip_imputation_cols: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    test_size: float
    val_size: float
    sample_pct: float
    # Делаем их опциональными, так как в проекте может быть только текст или только таблицы
    tabular: TabularDataConfig


# --- Блоки Обучения и Моделей ---

@dataclass
class ModelConfig:
    name: str
    version: str  # Версионирование (например: "1.0.0")
    # Гиперпараметры могут быть любыми (CatBoost, нейронка и т.д.),
    # поэтому оставляем гибкий словарь
    params: Dict[str, Any] = field(default_factory=dict)
    # Словарь параметров для поиска (например: {"depth": [4, 10], "lr": [0.01, 0.1]})
    optuna_search_space: Dict[str, Any] = field(default_factory=dict)

# --- Глобальные настройки обучения ---
@dataclass
class MlTrainingConfig:
    cv_folds: int
    early_stopping_rounds: int
    verbose: int


@dataclass
class DlTrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    early_stopping_patience: int
    optimizer: str
    scheduler: str
    momentum: float = 0.9

@dataclass
class OptunaConfig:
    n_trials: int
    direction: str  # "maximize" или "minimize"
    sampler: str    # "tpe" или "random"



@dataclass
class TrainingConfig:
    device: str
    num_workers: int
    pin_memory: bool
    ml: Optional[MlTrainingConfig] = None
    dl: Optional[DlTrainingConfig] = None
    optuna: Optional[OptunaConfig] = None


# --- Блоки Логирования ---

# --- MLflow ---
@dataclass
class MlflowConfig:
    tracking_uri: str
    experiments: Dict[str, str]
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PlotsConfig:
    fig_size: List[int]
    dpi: int
    font_size: int
    style: str
    grid: bool
    alpha: float
    spines_top: bool
    spines_right: bool


@dataclass
class LoggingConfig:
    level: str
    log_file: str
    mlflow: MlflowConfig
    plots: PlotsConfig

@dataclass
class SecurityConfig:
    allowed_keys: str = ""


# ============================================================
# ГЛАВНЫЙ КЛАСС (Оркестратор схемы)
# ============================================================

@dataclass
class AppConfig:
    """
    Главный класс конфигурации. Точно повторяет структуру config.yaml.
    """
    project_name: str
    run_name: str
    mode: str
    task_type: str
    seed: int
    loss_function: str
    metrics: List[str]

    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig

    env: str = "dev"
    database: Optional[DatabaseConfig] = None
    security: Optional[SecurityConfig] = None
    dvc: Optional[DvcConfig] = None