# 1. Берем легкий базовый образ Python
FROM python:3.10-slim

# 2. Копируем сверхбыстрый бинарник uv прямо из их официального образа
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Настраиваем переменные окружения
ENV PYTHONUNBUFFERED=1 \
    # Указываем uv собирать пакеты прямо в системный Python контейнера (без создания .venv)
    UV_SYSTEM_PYTHON=1 

WORKDIR /app

# 4. Копируем ТОЛЬКО файлы конфигурации зависимостей
# Это кэширует слой. Если код поменяется, а pyproject.toml нет — установка библиотек не будет запускаться заново
COPY pyproject.toml uv.lock* ./

# 5. УСТАНАВЛИВАЕМ ТОЛЬКО ПРОДАКШЕН ЗАВИСИМОСТИ
# --no-dev игнорирует блок [dependency-groups] (pytest, black)
# Мы не пишем --extra eda, поэтому блок [project.optional-dependencies] (matplotlib) тоже игнорируется
RUN uv sync --no-dev --no-install-project

# 6. Теперь копируем сам код проекта
COPY src/ src/
COPY api/ api/
COPY configs/ configs/

# 7. Устанавливаем сам проект (чтобы импорты работали корректно)
RUN uv sync --no-dev

# 8. Указываем порт и команду запуска (например, для FastAPI сервера)
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]