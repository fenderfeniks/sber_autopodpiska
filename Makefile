# Определяем интерпретатор Python и команду запуска тестов
PYTHON = python
PYTEST = python -m pytest

.PHONY: help venv install test test-unit test-integration run-train run-tune clean

# Команда по умолчанию (показывает список доступных команд)
help:
	@echo "Доступные команды для автоматизации проекта:"
	@echo "  make venv             - Создать виртуальное окружение venv"
	@echo "  make install          - Установить зависимости из requirements.txt"
	@echo "  make test             - Запустить ВСЕ тесты (unit + integration)"
	@echo "  make test-unit        - Запустить только быстрые юнит-тесты"
	@echo "  make test-integration - Запустить тяжелые интеграционные тесты"
	@echo "  make run-train        - Запустить обучение модели (mode=train)"
	@echo "  make run-tune         - Запустить подбор гиперпараметров (mode=tune)"
	@echo "  make clean            - Полная очистка проекта от кэша, pyc файлов и логов тестов"

# ============================================================
# ОКРУЖЕНИЕ И ЗАВИСИМОСТИ
# ============================================================
venv:
	$(PYTHON) -m venv .venv
	@echo "Окружение создано. Активация:"
	@echo "  Windows : .venv\\Scripts\\activate"
	@echo "  Linux/Mac: source .venv/bin/activate"

install:
	$(PYTHON) -m pip install --upgrade pip
    pip install -e ".[dev]"

# ============================================================
# ТЕСТИРОВАНИЕ (Pytest)
# ============================================================
test:
	$(PYTEST) tests/ -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v

# ============================================================
# ЗАПУСК ПРОЕКТА (Оркестратор main.py)
# ============================================================
run-train:
	$(PYTHON) main.py mode=train

run-tune:
	$(PYTHON) main.py mode=tune

# ============================================================
# ОЧИСТКА ПРОЕКТА
# ============================================================
clean:
	@echo "Очистка кэша и временных файлов..."
	rm -rf .pytest_cache outputs multirun .hydra catboost_info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -type f -delete
	@echo "Проект полностью очищен!"