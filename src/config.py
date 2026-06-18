from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    test_size: float = 0.2
    val_size: float = 0.3
    random_state: int = 42


@dataclass
# class FeatureConfig:
    # target_col: str = "SalePrice"
    # drop_cols: list = field(default_factory=list)
    # high_cardinality_threshold: int = 20


@dataclass
# class ModelConfig:
    # name: str = "catboost"
    # n_trials: int = 100
    # cv_folds: int = 5


@dataclass
class PlotsConfig:
    fig_size: tuple = (12, 6)
    dpi: int = 100
    font_size: int = 12
    style: str = "seaborn-v0_8-whitegrid"
    grid: bool = True
    alpha: float = 0.3
    spines_top: bool = False
    spines_right: bool = False

@dataclass
class PathsConfig:
    models_dir: str = "models"
    reports_dir: str = "reports"
    data_dir: str = "data"
    output_dir: str = "outputs"
    src_dir: str = "src"
    raw_dir: str = "raw"
    preprocessed_dir: str = "preprocessed"
    target_file_name: str = "ga_hits.csv"
    users_file_name: str = "ga_sessions.csv"


    def raw_users_path(self, root: Path) -> Path:
        return root / self.data_dir / self.raw_dir / self.users_file_name


    def raw_target_path(self, root: Path) -> Path:
        return root / self.data_dir / self.raw_dir / self.target_file_name


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    # features: FeatureConfig = field(default_factory=FeatureConfig)
    # model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    mlflow_experiment: str = "default"
    models_dir: str = "models"
    reports_dir: str = "reports"
    plots: PlotsConfig = field(default_factory=PlotsConfig)


def load_config(config_path: str | Path = "configs/config.yaml") -> ProjectConfig:
    """Загружает YAML и возвращает типизированный конфиг."""
    config_path = Path(config_path)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return ProjectConfig(
        data=DataConfig(**raw.get("data", {})),
        # features=FeatureConfig(**raw.get("features", {})),
        # model=ModelConfig(**raw.get("model", {})),
        paths=PathsConfig(**raw.get("paths", {})),
        mlflow_experiment=raw.get("mlflow", {}).get("experiment_name", "default"),
        models_dir=raw.get("paths", {}).get("models_dir", "models"),
        reports_dir=raw.get("paths", {}).get("reports_dir", "reports"),
        plots=PlotsConfig(**raw.get('plots', {})),
    )