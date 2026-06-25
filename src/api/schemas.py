from __future__ import annotations

import json
import logging
from pathlib import Path
from pydantic import BaseModel, create_model
from typing import Any, Optional, Union, List, Dict
from core.utils import load_hydra_config, PROJECT_ROOT

logger = logging.getLogger(__name__)

# 1. Динамически достаем версию из Hydra
try:
    cfg = load_hydra_config("config")
    MODEL_VERSION = cfg.model.version
    MODEL_DIR = cfg.paths.models_dir
except Exception:
    MODEL_DIR = 'models'
    MODEL_VERSION = "1.0.0"

# 3. Строим надежный абсолютный путь
SCHEMA_PATH = PROJECT_ROOT / MODEL_DIR / f"feature_schema_v{MODEL_VERSION}.json"

def build_dynamic_request_model(schema_path: Path):
    """
    Фабрика. Читает json со схемой колонок и возвращает готовый Pydantic класс.
    """
    if not schema_path.exists():
        logger.warning(
            f"[SCHEMAS] Файл схемы не найден: {schema_path}. "
            f"Обучите модель командой 'python main.py mode=train'. "
            f"Эндпоинт /predict работать не будет."
        )
        return create_model("PredictionRequest", placeholder=(str, "Модель еще не обучена"))

    with open(schema_path, "r") as f:
        feature_schema = json.load(f)

    # Маппинг типов Pandas/Numpy в стандартные типы Python
    type_mapping = {
        "int64": int,
        "int32": int,
        "float64": float,
        "float32": float,
        "object": str,
        "bool": bool,
        "category": str
    }

    fields: Dict[str, Any] = {}
    for col_name, dtype in feature_schema.items():
        python_type = type_mapping.get(dtype, Any)
        # Кортеж (Тип, ...) означает, что поле обязательное (required)
        fields[col_name] = (python_type, ...)

    # Динамически создаем класс с именем 'PredictionRequest'
    return create_model("PredictionRequest", **fields)


# ============================================================
# ГОТОВЫЕ ЭКСПОРТЫ ДЛЯ API
# ============================================================
# Этот класс сгенерируется в момент запуска uvicorn
PredictionRequest = build_dynamic_request_model(SCHEMA_PATH)

# Ответ от API всегда статичен, поэтому пишем его руками
class PredictionResponse(BaseModel):
    prediction: Union[float, int, str]
    probability: Optional[float] = None
    # Приятный бонус: отдаем весь массив вероятностей
    probabilities: Optional[List[float]] = None