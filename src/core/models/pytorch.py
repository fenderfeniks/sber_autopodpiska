from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from core.utils import PROJECT_ROOT
from .base import BaseModelWrapper

# Импортируем специфичные библиотеки внутри методов или через try/except,
# чтобы код не падал, если в проекте не установлен CatBoost или PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    PYTORCH_INSTALLED = True
except ImportError:
    PYTORCH_INSTALLED = False

logger = logging.getLogger(__name__)


# ============================================================
# 5. СОБСТВЕННЫЙ КЛАСС НЕЙРОНКИ
# ============================================================

class PyTorchWrapper(BaseModelWrapper):
    """
    Универсальная обертка для любых кастомных нейросетей на PyTorch.
    Сохраняет единый интерфейс fit(X, y) с классическими ML моделями.
    """

    def __init__(self, config: DictConfig, custom_nn: nn.Module = None):
        super().__init__(config)

        if not PYTORCH_INSTALLED: raise ImportError("PyTorch не установлен!")

        self.dl_cfg = self.cfg.training.dl
        self.device = torch.device(self.cfg.training.device if torch.cuda.is_available() else "cpu")

        # ПУНКТ 1: Безопасная инициализация
        if custom_nn is None:
            logger.warning("Архитектура custom_nn не передана! Используется дефолтная линейная заглушка.")
            out_features = 1 if self.task_type == 'regression' else 2
            # LazyLinear сам вычислит размер входа при первом батче
            self.model = nn.Sequential(
                nn.LazyLinear(64),
                nn.ReLU(),
                nn.LazyLinear(out_features)
            ).to(self.device)
        else:
            self.model = custom_nn.to(self.device)

        # 2. Оптимизатор и Loss (можно парсить из конфига)
        opt_name = self.dl_cfg.optimizer.lower()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.dl_cfg.learning_rate)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.dl_cfg.learning_rate,
                                             momentum=self.dl_cfg.momentum)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.dl_cfg.learning_rate)

        # Зависит от глобальной задачи
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        direction = getattr(self.cfg.training.optuna, 'direction', 'minimize') \
            if self.cfg.training.optuna else 'minimize'
        self.best_val_score = float('inf') if direction == 'minimize' else float('-inf')
        self._direction = direction

    def _prepare_dataloader(self, X: pd.DataFrame, y: pd.Series, shuffle: bool) -> DataLoader:
        """Внутренний метод: конвертирует Pandas в PyTorch DataLoader"""

        non_numeric_cols = X.select_dtypes(exclude=[np.number, bool]).columns
        if len(non_numeric_cols) > 0:
            raise TypeError(
                f"PyTorchWrapper принимает только числа! Найдены строковые/объектные колонки: {non_numeric_cols.tolist()}. "
                f"Используйте FeatureEngineer для их кодирования."
            )

        X_tensor = torch.tensor(X.astype(np.float32).values, dtype=torch.float32)

        if self.task_type == 'regression':
            y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
        else:
            y_tensor = torch.tensor(y.values, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=self.dl_cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            tracker=None) -> None:

        train_loader = self._prepare_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._prepare_dataloader(X_val, y_val, shuffle=False) if X_val is not None else None

        epochs = self.dl_cfg.epochs

        # ИСПРАВЛЕНО: Безопасное логирование параметров через внедренный tracker
        if tracker:
            tracker.log_params({"epochs": epochs, "batch_size": self.dl_cfg.batch_size, "device": str(self.device)})

        logger.info(f"Старт обучения PyTorch. Девайс: {self.device}, Эпох: {epochs}")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            # ТРЕНИРОВОЧНЫЙ ЦИКЛ
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            # ВАЛИДАЦИЯ
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        outputs = self.model(X_batch)
                        val_loss += self.criterion(outputs, y_batch).item() * X_batch.size(0)
                val_loss /= len(val_loader.dataset)

                if self._direction == 'minimize':
                    if val_loss < self.best_val_score:
                        self.best_val_score = val_loss
                else:
                    if val_loss > self.best_val_score:
                        self.best_val_score = val_loss

            # ЛОГИРОВАНИЕ ЭПОХИ (Через независимый ArtifactManager)
            if tracker:
                tracker.log_metrics({"train_loss": train_loss}, step=epoch)
                if val_loader:
                    tracker.log_metrics({"val_loss": val_loss}, step=epoch)

            logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Полноценный инференс для PyTorch"""
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)

            if self.task_type == 'regression':
                preds = outputs.squeeze(1)
            else:
                preds = torch.argmax(outputs, dim=1)

        return preds.cpu().numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Возвращает вероятности классов (только для задач классификации)."""
        if self.task_type == 'regression':
            raise NotImplementedError("Метод predict_proba недоступен для задачи регрессии.")

        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def save(self) -> str:
        """Нативное сохранение весов PyTorch модели (state_dict)."""
        file_name = f"{self.model_cfg.name}_v{self.model_cfg.model_version}.pt"
        save_path = PROJECT_ROOT / self.cfg.paths.models_dir / file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Веса PyTorch модели сохранены в {save_path}")
            # ИСПРАВЛЕНО: Убрали mlflow.log_artifact(str(save_path), artifact_path="models")

        return str(save_path)

    def load(self, load_path: str) -> None:
        """
        Загрузка весов PyTorch.
        ВНИМАНИЕ: Для загрузки кастомной нейросети (custom_nn),
        вы ОБЯЗАНЫ передать точно такую же архитектуру в конструктор
        PyTorchWrapper при инициализации. Иначе веса не совпадут.
        """

        if not PYTORCH_INSTALLED:
            raise ImportError("PyTorch не установлен!")

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Файл весов не найден: {load_path}")

        state_dict = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Веса PyTorch успешно загружены на устройство: {self.device}")

    @property
    def file_extension(self) -> str:
        return ".pt"

    def get_best_val_score(self, metric_name: str = None) -> float:
        """Единый интерфейс для Optuna для получения метрики валидации."""
        return getattr(self, 'best_val_score', 0.0)