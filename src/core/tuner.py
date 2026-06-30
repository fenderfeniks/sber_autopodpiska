from __future__ import annotations

import logging
import optuna
from omegaconf import DictConfig, OmegaConf
from optuna.samplers import TPESampler, RandomSampler

from src.core.pipeline import MLPipeline

logger = logging.getLogger(__name__)


class OptunaTuner:
    def __init__(self, cfg, project_root, tracker):
        self.project_root = project_root
        self.cfg = cfg
        self.optuna_cfg = cfg.training.optuna
        self.tracker = tracker

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
        import mlflow # Импортируем mlflow для управления контекстом

        def objective(trial):
            trial_cfg = OmegaConf.create(
                OmegaConf.to_container(self.cfg, resolve=True)
            )

            # 1. Генерируем параметры из пространства поиска
            search_space = trial_cfg.model.get("optuna_search_space", {})
            current_trial_params = {} # Собираем параметры текущего триала для логирования
            
            for param_name, space_info in search_space.items():
                if isinstance(space_info, (dict, DictConfig)):
                    p_type = space_info.get("type", "float")
                    bounds = space_info.get("bounds", [])
                    is_log = space_info.get("log", False)
                    
                    if len(bounds) == 2:
                        if p_type == "int":
                            value = trial.suggest_int(param_name, bounds[0], bounds[1])
                        else:
                            value = trial.suggest_float(param_name, bounds[0], bounds[1], log=is_log)
                        
                        OmegaConf.update(trial_cfg, f"model.params.{param_name}", value)
                        current_trial_params[param_name] = value

            # === СТАРТ ВЛОЖЕННОГО РАНА В MLFLOW ===
            # run_name делаем коротким, например "trial_0", "trial_1"
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                
                # Логируем параметры текущей попытки Optuna в MLflow
                mlflow.log_params(current_trial_params)

                # 2. Запускаем стандартный пайплайн с новыми параметрами
                pipeline = MLPipeline(cfg=trial_cfg, project_root=self.project_root, tracker=self.tracker)
                pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=False, use_tracker=False)

                # 3. Получаем метрику качества
                target_metric = trial_cfg.metrics[0]
                score = pipeline.model.get_best_val_score(target_metric)
                
                # Логируем итоговую метрику триала в MLflow
                mlflow.log_metric(f"val_{target_metric}", score)

            return score

        # Запуск без внешних конфликтующих коллбэков
        study = optuna.create_study(
            direction=self.optuna_cfg.direction,
            sampler=self._get_sampler()
        )
        study.optimize(objective, n_trials=self.optuna_cfg.n_trials)

        logger.info(f"Лучшие параметры: {study.best_params}")
        return study.best_params   