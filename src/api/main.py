from __future__ import annotations

import time
import logging
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
from prometheus_client import make_asgi_app, Counter, Histogram

# Pydantic схемы (предполагаем, что они лежат в api/schemas.py)
from api.schemas import PredictionRequest, PredictionResponse
from api.dependencies import verify_api_key, get_ml_model, get_preprocessor
from src.core.utils import load_hydra_config, PROJECT_ROOT

# Импорты движка
from src.core.models import get_model

logger = logging.getLogger(__name__)

# === 1. ОБЪЯВЛЯЕМ МЕТРИКИ ДЛЯ PROMETHEUS ===
# Счетчик всех предсказаний (с разделением по предсказанному классу)
PREDICTION_COUNTER = Counter(
    'ml_predictions_total', 
    'Total number of predictions made',
    ['predicted_class'] # Лейбл, по которому мы будем строить пироги в Графане
)

# Гистограмма времени ответа (SLA мониторинг)
INFERENCE_LATENCY = Histogram(
    'ml_inference_latency_seconds', 
    'Time spent processing the prediction',
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0) # Корзины для P95 / P99
)

# Гистограмма уверенности сети (только для классификации)
PREDICTION_CONFIDENCE = Histogram(
    'ml_prediction_confidence', 
    'Probability of the predicted class',
    buckets=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
)

# ============================================================
# 1. ЗАГРУЗКА КОНФИГА ДЛЯ МЕТАДАННЫХ (Global Level)
# ============================================================
try:
    # Загружаем конфиг один раз при старте модуля
    cfg = load_hydra_config("config")
    API_TITLE = cfg.project_name
    MODEL_VERSION = cfg.model.model_version
    API_VERSION = MODEL_VERSION  # Поменять если добавить в конфиг версию API
    API_DESC = f"Микросервис инференса. Модель: {cfg.model.name}"
    HAS_CONFIG = True
except Exception as e:
    logger.warning(f"Не удалось загрузить конфиг для метаданных: {e}")
    API_TITLE = "ML Production API"
    MODEL_VERSION = "1.0.0"
    API_VERSION = "1.0.0"
    API_DESC = "Ожидание загрузки модели..."
    HAS_CONFIG = False

# ============================================================
# 2. ЗАГРУЗКА ЗАВИСИМОСТЕЙ (Lifespan)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Загрузка моделей в память...")
    if not HAS_CONFIG:
        logger.error("Запуск без конфигов. Инференс будет возвращать ошибку 503.")
        app.state.model = None
        app.state.preprocessor = None
        yield
        return

    try:
        # ПУТИ ИЗ КОНФИГА ОТ КОРНЯ ПРОЕКТА
        models_dir = PROJECT_ROOT / cfg.paths.models_dir

        # Загрузка препроцессора
        prep_path = models_dir / f"preprocessing_v{cfg.model.version}.pkl"
        app.state.preprocessor = joblib.load(prep_path)

        # Загрузка модели
        model_wrapper = get_model(cfg)
        model_path = models_dir / f"{cfg.model.name}_v{cfg.model.version}{model_wrapper.file_extension}"

        model_wrapper.load(model_path)
        app.state.model = model_wrapper

        logger.info("API успешно инициализировано.")
    except Exception as e:
        logger.error(f"Ошибка загрузки весов: {e}")
        # Не роняем процесс, позволяем Swagger UI запуститься
        app.state.model = None
        app.state.preprocessor = None

    yield

    logger.info("Остановка сервера. Очистка памяти...")
    app.state.model = None
    app.state.preprocessor = None


# ============================================================
# 3. ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ
# ============================================================
app = FastAPI(
    title=API_TITLE,
    description=API_DESC,
    version=API_VERSION,
    lifespan=lifespan
)
# === 2. ОТКРЫВАЕМ ЭНДПОИНТ ДЛЯ PROMETHEUS ===
# По адресу /metrics Prometheus будет забирать наши данные
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ============================================================
# 4. ГЛАВНЫЙ ЭНДПОИНТ ИНФЕРЕНСА
# ============================================================
# Защищаем эндпоинт ключом через dependencies=[Depends(verify_api_key)]
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest,
            api_key: str = Depends(verify_api_key),
            model_wrapper=Depends(get_ml_model),      # ИСПОЛЬЗУЕМ ЕДИНЫЙ ИСТОЧНИК
            preprocessor=Depends(get_preprocessor)):
    """
    ВАЖНО: Функция объявлена как `def`, а не `async def`. 
    FastAPI запустит её в Threadpool, чтобы Pandas не заблокировал сервер.
    """
    if model_wrapper is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена или отключена.")

    start_time = time.perf_counter()

    try:
        # 1. Pydantic-схему (request) превращаем в DataFrame (1 строка)
        df_input = pd.DataFrame([request.model_dump()])

        # 2. Очистка данных (наш TabularPreprocessor + FeatureEngineer)
        df_clean = preprocessor.transform(df_input)

        # 3. Предсказание
        pred_value = model_wrapper.predict(df_clean)[0]
        if isinstance(pred_value, np.generic):
            pred_value = pred_value.item()

        # Опционально: вероятность (если это бинарная классификация)
        probability = None
        probabilities = None
        try:
            if hasattr(model_wrapper, 'predict_proba'):
                proba_array = model_wrapper.predict_proba(df_clean)[0]
                probability = float(np.max(proba_array))
                probabilities = proba_array.tolist()
                # Записываем метрику уверенности модели
                PREDICTION_CONFIDENCE.observe(probability)

        except NotImplementedError:
            pass

        # === 3. ЗАПИСЫВАЕМ БИЗНЕС-МЕТРИКИ ===
        # Увеличиваем счетчик предсказаний, помечая, какой класс выдала модель
        PREDICTION_COUNTER.labels(predicted_class=str(pred_value)).inc()
        
        # Считаем, сколько времени занял инференс, и отправляем в гистограмму
        process_time = time.perf_counter() - start_time
        INFERENCE_LATENCY.observe(process_time)
        # ====================================
        
        return PredictionResponse(
            prediction=pred_value,
            probability=probability,
            probabilities=probabilities
        )
    except Exception as e:
        logger.error(f"Ошибка во время инференса: {str(e)} | Данные: {request.model_dump()}")
        # Возвращаем клиенту понятную 500 ошибку, а не просто "Internal Server Error"
        raise HTTPException(status_code=500, detail="Ошибка обработки предсказания внутри модели.")


# ============================================================
# 5. ЭНДПОИНТ ПРОВЕРКИ ЗДОРОВЬЯ (Healthcheck)
# ============================================================
@app.get("/health", tags=["System"])
def health_check(request: Request):
    if not request.app.state.model:
        return {"status": "degraded", "detail": "Модель не загружена"}
    return {"status": "ok", "model_version": MODEL_VERSION}