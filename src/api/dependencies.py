from __future__ import annotations

from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from core.utils import load_hydra_config

# Ждем заголовок 'X-API-Key' в запросе
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def _parse_allowed_keys(cfg) -> set[str]:
    """Парсит строку ключей из конфига в множество."""
    raw = cfg.security.allowed_keys if cfg.security else ""
    if not raw:
        raise RuntimeError(
            "cfg.security.allowed_keys пуст. "
            "Проверьте переменную API_ALLOWED_KEYS в .env"
        )
    return {key.strip() for key in raw.split(",") if key.strip()}

def verify_api_key(api_key: str = Security(api_key_header)):
    """Проверка ключа. В dev-режиме пропускает любой ключ."""
    cfg = load_hydra_config()

    if cfg.env == "dev":
        return api_key

    valid_keys = _parse_allowed_keys(cfg)

    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Неверный API ключ")

    return api_key

def get_ml_model(request: Request):
    """
    Зависимость: достает модель из памяти сервера.
    Это избавляет нас от использования глобальных переменных.
    """
    model = request.app.state.model
    if not model:
        raise HTTPException(status_code=503, detail="Модель еще загружается...")
    return model

def get_preprocessor(request: Request):
    return request.app.state.preprocessor