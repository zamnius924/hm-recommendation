from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

with DAG(

    # ------------------------------- Параметры DAG ------------------------------ #
    dag_id='recommend_dag', # название DAG
    description='Train recommendations model', # описание DAG
    schedule_interval=timedelta(days=1), # как часто запускать DAG
    start_date=datetime(2026, 1, 1), # дата начала DAG
    catchup=False, # запускать ли DAG за пропущенные интервалы
    
    # параметры по умолчанию для тасок
    default_args={
        'depends_on_past': False, # если прошлые запуски упали, надо ли ждать их успеха
        'email_on_failure': False, # писать ли при провале
        'email_on_retry': False, # писать ли при автоматическом перезапуске по провалу
        'retries': 3, # сколько раз пытаться запустить, далее помечать как failed
        'retry_delay': timedelta(minutes=5) # сколько ждать между перезапусками
    }

) as dag:
    
    # ----------------------------------- Таски ---------------------------------- #
    # Таска 1: определение окон для тренировочной, валидационной и тестовой выборок
    t1 = BashOperator(
        task_id='load_db',
        bash_command='python pipeline/run_loader.py'
    )

    # Таска 2: создание кандидатов ALS и фичей
    t2 = BashOperator(
        task_id='feature_engineering',
        bash_command='python pipeline/run_feature_engineering.py'
    )

    # Таска 3: обучение модели LTR
    t3 = BashOperator(
        task_id='fit_ltr',
        bash_command='python pipeline/run_ltr_fit.py'
    )

    # Таска 4: генерация рекомендация и загрузка их в БД
    t4 = BashOperator(
        task_id='generate_recommendations',
        bash_command='python pipeline/run_recommendations.py'
    )

    # ------------------------------------ DAG ----------------------------------- #
    # Определение порядка выполнения тасок
    t1 >> t2 >> t3 >> t4