from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from omegaconf import DictConfig

class BaseModelWrapper(ABC):
    def __init__(self, config: DictConfig, project_root):
        self.cfg = config
        self.PROJECT_ROOT = project_root
        self.model_cfg = config.model
        self.task_type = config.task_type
        self.model = None

    @property
    @abstractmethod
    def file_extension(self) -> str: pass

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, tracker=None) -> None:
        """
        Обучение модели.
        :param tracker: Экземпляр ArtifactManager для логирования по эпохам (опционально)
        """
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray: pass

    @abstractmethod
    def get_best_val_score(self, metric_name: str) -> float: pass

    @abstractmethod
    def save(self) -> str: pass

    @abstractmethod
    def load(self, load_path: str) -> None: pass