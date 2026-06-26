from __future__ import annotations

from pathlib import Path
from omegaconf import DictConfig
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.config_store import ConfigStore
from src.core.config_schema import AppConfig
import functools

def get_project_root() -> Path:
    """
    Динамически находит корень проекта, поднимаясь по директориям вверх,
    пока не найдет папку 'configs' или 'pyproject.toml'.
    """
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "configs").exists() or (parent / "pyproject.toml").exists():
            return parent

    # Fallback на случай нестандартной структуры
    return Path(__file__).resolve().parent.parent.parent


# ГЛОБАЛЬНАЯ КОНСТАНТА КОРНЯ ПРОЕКТА
PROJECT_ROOT = get_project_root()


@functools.lru_cache(maxsize=1)
def load_hydra_config(config_name: str = "config") -> DictConfig:
    """Загружает конфиг один раз, валидирует по схеме AppConfig и кеширует его."""
    
    # Регистрируем схему типов в глобальном хранилище Hydra
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=AppConfig)

    # Инициализируем только если Hydra еще не запущена в текущем процессе
    if not GlobalHydra.instance().is_initialized():
        # PROJECT_ROOT должен быть доступен (мы берем его из окружения процесса)
        from pathlib import Path
        import os
        
        # На всякий случай рассчитываем путь от корня, если cwd сдвинут
        abs_config_path = str(Path(os.getcwd()) / "configs")
        initialize_config_dir(version_base=None, config_dir=abs_config_path)

    return compose(config_name=config_name)


def clear_config_cache():
    """Сбрасывает кэш конфига (использовать только в Unit-тестах)."""
    load_hydra_config.cache_clear()
    GlobalHydra.instance().clear()