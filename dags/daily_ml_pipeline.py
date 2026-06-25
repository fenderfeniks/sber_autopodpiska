from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Базовые настройки для всех задач в этом графе (DAG)
default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2, # Если упадет, попробует еще 2 раза
    'retry_delay': timedelta(minutes=5),
}

# Путь до твоего проекта на сервере
PROJECT_PATH = "/opt/ml_project"

# Определение графа
with DAG(
    'daily_churn_model_retraining',
    default_args=default_args,
    description='Ежедневное дообучение и оценка модели',
    schedule_interval='0 3 * * *', # Запуск каждый день в 3:00 ночи
    start_date=datetime(2026, 6, 20),
    catchup=False, # Не запускать за пропущенные дни в прошлом
    tags=['ml', 'churn'],
) as dag:

    # Задача 1: Обучение модели
    # Hydra сама подхватит mode=train и сохранит артефакты
    train_model = BashOperator(
        task_id='train_model',
        bash_command=f'cd {PROJECT_PATH} && python src/main.py mode=train',
    )

    # Задача 2: Оценка на отложенной выборке
    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command=f'cd {PROJECT_PATH} && python src/main.py mode=evaluate',
    )

    # Задача 3: (Опционально) Инференс на новых данных
    batch_inference = BashOperator(
        task_id='batch_inference',
        bash_command=f'cd {PROJECT_PATH} && python src/main.py mode=inference',
    )

    # Порядок выполнения (Зависимости)
    # Если train_model упадет, следующие задачи НЕ запустятся
    train_model >> evaluate_model >> batch_inference