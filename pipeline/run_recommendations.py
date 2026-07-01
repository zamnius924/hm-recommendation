import json
import pandas as pd

from catboost import CatBoostRanker
from scripts.generate_als_candidates import generate_als_candidates
from scripts.generate_features import generate_features
from scripts.generate_scores import generate_scores
from scripts.load_db import load_db
from scripts.load_window_config import load_window_config
from scripts.paths import DATA_PROD_DIR, MODELS_PROD_DIR, MODELS_PROD_CONFIG_DIR

def run_recommendations():

    # ------------------------------ Загрузка данных ----------------------------- #
    # Длины окон для формирования таргета и фичей
    window_length = load_window_config()

    # Оптимальные параметры ALS
    with open(file=MODELS_PROD_CONFIG_DIR / 'als_best_params.json', mode='r') as file:
        als_best_params = json.load(file)

    # Загрузка модели
    model = CatBoostRanker()
    model.load_model(MODELS_PROD_DIR / 'ltr_model.cbm')

    # Подключение к БД
    con = load_db()


    # --------------------- Дата отсечения и окно для данных --------------------- #
    # Отсечка – дата, на которую строим рекомендации (последняя дата в данных)
    cutoff_date = con.execute("""
        SELECT MAX(t_dat)
        FROM transactions
    """).fetchone()[0]

    # Границы окна
    inference_dates = {
        'feature_window_end': cutoff_date,
        'feature_window_start': cutoff_date - window_length['feature']
    }


    # -------------------------- Построение рекомендаций ------------------------- #
    # Прогноз ALS
    als_candidates = generate_als_candidates(con,
                                             inference_dates,
                                             als_best_params,
                                             target=False)

    # Создание фичей
    con = generate_features(con,
                            inference_dates,
                            als_candidates,
                            target=False,
                            return_con=True)

    # Прогноз скоров на основе LTR
    df_rec = generate_scores(con, model, 5000)


    # -------------------------- Сохранение и отключение ------------------------- #
    # Сохранение рекомендаций
    df_rec.to_parquet(DATA_PROD_DIR / 'df_rec.parquet',
                      engine='pyarrow', index=False)
    
    # Отключение от БД
    con.close()

if __name__ == '__main__':
    run_recommendations()