import json
import pandas as pd

from catboost import CatBoostRanker
from scripts.ltr.generate_pool import generate_pool
from scripts.evaluation.map_at_k import map_at_k
from scripts.utils.paths import DATA_PROD_DIR, MODELS_PROD_DIR, MODELS_PROD_CONFIG_DIR

def run_ltr_fit():

    # ------------------------------ Загрузка данных ----------------------------- #
    # Оптимальные параметры LTR
    with open(file=MODELS_PROD_CONFIG_DIR / 'ltr_best_params.json', mode='r') as file:
        ltr_best_params = json.load(file)
    
    # Датасеты
    df_train = pd.read_parquet(DATA_PROD_DIR / 'df_train.parquet', engine='pyarrow')
    df_test = pd.read_parquet(DATA_PROD_DIR / 'df_test.parquet', engine='pyarrow')


    # ---------------------------- Обучение LTR-модели --------------------------- #
    # Создание пулов
    pool_train = generate_pool(df_train)
    pool_test = generate_pool(df_test)

    # Обучение CatBoost
    model = CatBoostRanker(**ltr_best_params['model'])
    model.fit(
        pool_train,
        eval_set=pool_test,
        **ltr_best_params['fit'])
    

    # ------------------------ Проверка на качество модели ----------------------- #
    # MAP@12 на тестовой выборке
    score = map_at_k(df_test, model.predict(pool_test))

    if score < 0.03:
        raise RuntimeError('Model quality degraded')
    

    # -------------------------------- Сохранение -------------------------------- #
    model.save_model(MODELS_PROD_DIR / 'ltr_model.cbm')


if __name__ == '__main__':
    run_ltr_fit()
