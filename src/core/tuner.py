from __future__ import annotations

import logging
import optuna
from omegaconf import DictConfig, OmegaConf, ListConfig
from optuna.samplers import TPESampler, RandomSampler

from core.pipeline import MLPipeline

logger = logging.getLogger(__name__)


class OptunaTuner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.optuna_cfg = cfg.training.optuna

    def _get_sampler(self):
        name = self.optuna_cfg.sampler.lower()
        if name == "tpe":
            return TPESampler()
        elif name == "random":
            return RandomSampler()
        else:
            logger.warning(f"Неизвестный sampler '{name}', используется TPE.")
            return TPESampler()


    def tune(self, X_train, y_train, X_val, y_val, tracker=None):
        logger.info(f"Старт Optuna (Trials: {self.optuna_cfg.n_trials})")

        def objective(trial):
            # Конвертируем в plain dict (resolve=True раскрывает interpolation-ссылки)
            trial_cfg = OmegaConf.create(
                OmegaConf.to_container(self.cfg, resolve=True)
            )

            # 1. Читаем диапазоны из конфига и генерируем параметры
            search_space = trial_cfg.model.get("optuna_search_space", {})
            for param_name, bounds in search_space.items():
                if isinstance(bounds, (list, ListConfig)) and len(bounds) == 2:
                    # Если оба числа целые - suggest_int, иначе suggest_float
                    if all(isinstance(x, int) for x in bounds):
                        value = trial.suggest_int(param_name, bounds[0], bounds[1])
                    else:
                        value = trial.suggest_float(param_name, bounds[0], bounds[1], log=True)

                    OmegaConf.update(trial_cfg, f"model.params.{param_name}", value)

            # 2. Запускаем стандартный пайплайн с новыми параметрами
            # Выключаем сохранение модели при переборе, чтобы не забить диск
            pipeline = MLPipeline(trial_cfg)
            pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=False, use_tracker=False)

            # 3. Возвращаем метрику качества
            target_metric = trial_cfg.metrics[0]
            return pipeline.model.get_best_val_score(target_metric)

        # Optuna логирует свои шаги автоматически, но мы берем коллбэк у нашего Менеджера
        callbacks = tracker.get_optuna_callback(metric_name="val_score") if tracker else []

        study = optuna.create_study(
            direction=self.optuna_cfg.direction,
            sampler=self._get_sampler()
        )
        study.optimize(objective, n_trials=self.optuna_cfg.n_trials, callbacks=callbacks)

        logger.info(f"Лучшие параметры: {study.best_params}")
        return study.best_params