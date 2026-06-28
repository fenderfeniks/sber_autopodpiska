from __future__ import annotations

import logging
import json
import joblib
from sklearn.pipeline import Pipeline
from omegaconf import DictConfig, OmegaConf

# Внутренние импорты
from core.features import TabularPreprocessor, FeatureEngineer
from core.models import get_model
from core.utils import PROJECT_ROOT
from core.metrics import calculate_metrics
from core.artifacts import ArtifactManager

logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Главный оркестратор.
    Сохраняет состояние обучения внутри себя для последующего инференса.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.preprocessor = None
        self.model = None
        # Инициализируем единый менеджер артефактов
        self.tracker = ArtifactManager(cfg)

    def _get_preprocessor(self) -> Pipeline:
        """Собирает Scikit-Learn пайплайн для обработки фичей."""
        return Pipeline([
            ('engineer', FeatureEngineer(self.cfg)),
            ('cleaner', TabularPreprocessor(self.cfg))     
        ])

    def train(self, X_train, y_train, X_val=None, y_val=None,
              save_artifacts: bool = True, use_tracker: bool = True):
        """Полный цикл обучения: препроцессинг -> модель."""
        logger.info("Начало работы ML-пайплайна...")

        # 1. Собираем и обучаем препроцессор
        self.preprocessor = self._get_preprocessor()
        logger.info("Применение препроцессинга к Train...")

        # Сначала фитим сам пайплайн, чтобы sklearn зафиксировал состояние "fitted"
        self.preprocessor.fit(X_train, y_train)

        # Теперь трансформируем данные
        X_train_clean = self.preprocessor.transform(X_train)

        # 2. Обрабатываем валидацию
        X_val_clean = None
        if X_val is not None:
            logger.info("Применение препроцессинга к Validation...")
            X_val_clean = self.preprocessor.transform(X_val)

        # 3. Инициализация и обучение модели
        self.model = get_model(self.cfg)

        active_tracker = self.tracker if use_tracker else None
        # ПЕРЕДАЕМ ТРЕКЕР В МОДЕЛЬ (Dependency Injection)
        self.model.fit(
            X_train_clean, y_train,
            X_val=X_val_clean, y_val=y_val,
            tracker=active_tracker
        )

        # === ФИНАЛЬНАЯ ОЦЕНКА МЕТРИК ===
        if X_val is not None and y_val is not None:
            logger.info("Расчет финальных бизнес-метрик на Validation...")

            # 1. Предсказываем классы / значения
            y_val_pred = self.model.predict(X_val_clean)

            # 2. Пытаемся получить вероятности (если это классификация)
            y_val_prob = None
            if self.cfg.task_type in ['binary', 'multiclass']:
                try:
                    if hasattr(self.model, 'predict_proba'):
                        y_val_prob = self.model.predict_proba(X_val_clean)
                except NotImplementedError:
                    pass  # Для кастомных моделей, где predict_proba не реализован

            # 3. Считаем все метрики разом
            final_metrics = calculate_metrics(y_val, y_val_pred, self.cfg.task_type, y_prob=y_val_prob)

            # 4. Логируем через менеджер артефактов
            if active_tracker:
                final_metrics_to_log = {f"final_val_{k}": v for k, v in final_metrics.items()}
                active_tracker.log_metrics(final_metrics_to_log)

            for m_name, m_value in final_metrics.items():
                logger.info(f"Final {m_name.upper()}: {m_value:.4f}")

        # 4. Сохранение артефактов
        if save_artifacts:
            logger.info("Сохранение артефактов...")

            # Сохраняем конфиг в трекер
            self.tracker.log_dict(OmegaConf.to_container(self.cfg, resolve=True), "config.json")

            # Подготавливаем папку
            models_dir = PROJECT_ROOT / self.cfg.paths.models_dir
            models_dir.mkdir(parents=True, exist_ok=True)

            # --- ПРАВКА 1: Берём новые версии из раздельных конфигов ---
            prep_ver = self.cfg.data.tabular.preprocessing_version
            feat_ver = self.cfg.data.tabilar.features_version
            model_ver = self.cfg.model.model_version

            # 1. Сохраняем схему фичей для FastAPI
            feature_types = X_train.dtypes.apply(lambda x: x.name).to_dict()
            schema_name = f"feature_schema_v{feat_ver}.json"
            schema_path = models_dir / schema_name

            with open(schema_path, "w") as f:
                json.dump(feature_types, f, indent=4)

            self.tracker.log_dict(feature_types, schema_name, "schemas")
            logger.info(f"Схема фичей сохранена in {schema_path}")

            # 2. Сохраняем препроцессор (версия из data.tabular)
            prep_path = models_dir / f"preprocessing_v{prep_ver}.pkl"
            joblib.dump(self.preprocessor, prep_path)
            self.tracker.log_artifact(str(prep_path), "preprocessing")

            # 3. Сохраняем модель (версия из model)
            model_path = self.model.save() # Внутри обертки CatBoostWrapper подхватится модель_версия
            self.tracker.log_artifact(model_path, "models")

            # 4. Логируем параметры из конфига
            if hasattr(self.cfg.model, 'params'):
                self.tracker.log_params(
                    OmegaConf.to_container(self.cfg.model.params, resolve=True)
                )

        return self.model

    def predict(self, X):
        """Экспортный метод для инференса (используется в FastAPI и тестах)."""
        if self.preprocessor is None or self.model is None:
            raise ValueError("Пайплайн еще не обучен! Вызовите метод train() перед predict().")

        # Сквозной инференс: чистим -> прогнозируем
        X_clean = self.preprocessor.transform(X)
        return self.model.predict(X_clean)

    def load(self) -> "MLPipeline":
        """
        Восстанавливает состояние пайплайна из сохраненных артефактов на диске.
        Использует раздельные версии компонентов.
        """
        # --- ПРАВКА 2: Извлекаем раздельные версии для загрузки ---
        prep_ver = self.cfg.data.tabular.preprocessing_version
        model_ver = self.cfg.model.model_version
        
        logger.info(f"Загрузка артефактов (Prep: v{prep_ver}, Model: v{model_ver})...")

        models_dir = PROJECT_ROOT / self.cfg.paths.models_dir
        if not models_dir.exists():
            raise FileNotFoundError(f"Директория с моделями не найдена: {models_dir}")

        # 1. Загрузка препроцессора по его личной версии
        prep_path = models_dir / f"preprocessing_v{prep_ver}.pkl"
        if not prep_path.exists():
            raise FileNotFoundError(f"Препроцессор не найден: {prep_path}")
        self.preprocessor = joblib.load(prep_path)
        logger.info("Препроцессор успешно загружен.")

        # 2. Инициализация архитектуры модели и загрузка весов по версии модели
        self.model = get_model(self.cfg)
        model_path = models_dir / f"{self.cfg.model.name}_v{model_ver}{self.model.file_extension}"
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
            
        self.model.load(str(model_path))

        logger.info("Пайплайн успешно восстановлен и готов к инференсу!")
        return self