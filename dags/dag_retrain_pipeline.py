"""
DAG: retrain_pipeline

Назначение:
    Полный цикл переобучения модели: train -> evaluate.
    Деплой НЕ автоматический — после evaluate инженер должен изучить
    метрики в MLflow UI и вручную запустить отдельный DAG `deploy_model`
    через Airflow UI ("Trigger DAG"), передав версию модели для деплоя.

    Это намеренное архитектурное решение: на retrain-пайплайнах с
    бизнес-метриками (а не просто accuracy) автоматический деплой по
    порогу — частая причина инцидентов, если метрика на тесте улучшилась
    но модель деградировала на специфичном сегменте. Человек в контуре
    дешевле инцидента в проде.

Архитектура (DockerOperator):
    Airflow здесь — чистый оркестратор. Он не импортирует и не запускает
    ML-код напрямую (airflow-образ намеренно лёгкий, без mlflow/optuna/
    catboost). Каждая ML-таска поднимает ОТДЕЛЬНЫЙ контейнер из образа
    sber_autopodpiska-train:latest (core + train extra, см. pyproject.toml)
    через хостовый docker.sock, выполняет в нём `python -m main mode=...`
    и контейнер удаляется после завершения.

    train/evaluate-контейнеры подключены к сети ml_network, чтобы видеть
    mlflow:5000 и postgres:5432 по именам сервисов из docker-compose.yml.
    Volume models_data общий с сервисом api — train пишет туда веса,
    api их читает.

Запуск:
    Вручную через Airflow UI, либо по расписанию (например еженедельно)
    если команда уверена в стабильности пайплайна данных.
"""
from __future__ import annotations

import os
import pendulum

from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from docker.types import Mount

TRAIN_IMAGE = "sber_autopodpiska-train:latest"
# Имя сети Docker Compose генерирует как <имя_папки_проекта>_<имя_сети>.
# Если compose-файл лежит в configs/deploy/, имя проекта по умолчанию = "deploy",
# поэтому сеть = "deploy_ml_network". Если запускаешь compose с -p/COMPOSE_PROJECT_NAME,
# или имя папки другое — проверь `docker network ls` и поправь константу ниже.
DOCKER_NETWORK = "deploy_ml_network"
DOCKER_URL = "unix://var/run/docker.sock"

# Извлекаем переменные окружения, проброшенные в airflow-scheduler
DB_USER = os.environ.get("DB_USER", "admin")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "admin")
DB_NAME = os.environ.get("DB_NAME", "mlflow_db")

default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}


def print_mlflow_reminder(**context):
    print(
        "Train + evaluate завершены.\n"
        "Проверьте метрики в MLflow UI (http://localhost:5000).\n"
        "Если всё устраивает, запустите DAG 'deploy_model' вручную."
    )


def common_docker_kwargs(task_id: str) -> dict:
    """Общие параметры DockerOperator, чтобы не дублировать в каждой таске."""
    return dict(
        task_id=task_id,
        image=TRAIN_IMAGE,
        api_version="auto",
        auto_remove="success",
        docker_url=DOCKER_URL,
        network_mode=DOCKER_NETWORK,
        mount_tmp_dir=False,
        mounts=[
            # Train пишет сюда веса/препроцессор/feature_schema —
            # тот же volume, что смонтирован в сервис api.
            Mount(source="deploy_models_data", target="/app/models", type="volume"),
        ],
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            # Прокидываем переменные для oc.env резолвера в Hydra внутри train-контейнера
            "POSTGRES_HOST": "postgres", # Имя сервиса БД в сети docker-compose
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": DB_USER,
            "POSTGRES_PASSWORD": DB_PASSWORD,
            "POSTGRES_DB": DB_NAME,
        },
    )


# ============================================================
# ПАЙПЛАЙН ОБУЧЕНИЯ
# ============================================================
with DAG(
    dag_id="retrain_pipeline",
    description="Пайплайн переобучения: Train -> Evaluate",
    default_args=default_args,
    schedule=None,
    start_date=pendulum.datetime(2026, 6, 1, tz="UTC"),
    catchup=False,
    tags=["ml", "training", "manual-gate", "docker-operator"],
) as dag:

    train_model = DockerOperator(
        **common_docker_kwargs("train_model"),
        command="mode=train",
    )

    evaluate_model = DockerOperator(
        **common_docker_kwargs("evaluate_model"),
        command="mode=evaluate",
    )

    manual_gate_reminder = PythonOperator(
        **{"task_id": "manual_gate_reminder", "python_callable": print_mlflow_reminder}
    )

    train_model >> evaluate_model >> manual_gate_reminder


# ============================================================
# ПАЙПЛАЙН ДЕПЛОЯ
# ============================================================
with DAG(
    dag_id="deploy_model",
    description="Деплой модели в прод. Запускается вручную.",
    default_args=default_args,
    schedule=None,
    start_date=pendulum.datetime(2026, 6, 1, tz="UTC"),
    catchup=False,
    tags=["ml", "deploy", "manual-only"],
) as deploy_dag:

    # Модель уже лежит в volume models_data (её туда положил train-контейнер
    # на предыдущем шаге retrain_pipeline) — она физически видна сервису api
    # (тот же volume смонтирован в docker-compose.yml). Единственное, что
    # нужно для "подхвата" новой версии — перечитать веса в FastAPI lifespan,
    # а самый простой способ это сделать без правки кода api — перезапустить
    # сам контейнер. Имя контейнера фиксировано через container_name: sber_api
    # в docker-compose.yml, поэтому достаточно простого `docker restart` через
    # проброшенный docker.sock — без docker compose CLI внутри airflow-образа.
    #
    # Если впоследствии добавишь POST /reload в FastAPI (читающий веса заново
    # без даунтайма) — эту BashOperator-таску можно будет заменить на простой
    # HTTP-запрос к api, без рестарта контейнера.
    deploy_artifacts = BashOperator(
        task_id="deploy_artifacts",
        bash_command=(
            "echo 'Перезапуск api для подхвата новой версии модели...' && "
            "docker restart sber_api && "
            "echo 'Деплой успешно завершен!'"
        ),
    )

    deploy_artifacts