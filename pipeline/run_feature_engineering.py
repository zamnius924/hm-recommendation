import json

from scripts.generate_als_candidates import generate_als_candidates
from scripts.generate_features import generate_features
from scripts.load_db import load_db
from scripts.paths import DATA_PROD_DIR, MODELS_PROD_CONFIG_DIR

def run_feature_engineering():

    # ---------------------------- Загрузка параметров --------------------------- #
    # Границы окон
    with open(file=DATA_PROD_DIR / 'split_dates.json', mode='r') as file:
        dates = json.load(file)

    # Оптимальные параметры ALS
    with open(MODELS_PROD_CONFIG_DIR / 'als_best_params.json', mode='r') as file:
        als_best_params = json.load(file)

    # Покдючение к БД
    con = load_db()

    # ----------------------------- Обработка данных ----------------------------- #
    # Кандидаты ALS
    als_candidates_train = generate_als_candidates(con, dates['train'], als_best_params)
    als_candidates_valid = generate_als_candidates(con, dates['valid'], als_best_params)
    als_candidates_test = generate_als_candidates(con, dates['test'], als_best_params)

    # Создание фичей
    df_train = generate_features(con, dates['train'], als_candidates_train)
    df_valid = generate_features(con, dates['valid'], als_candidates_valid)
    df_test = generate_features(con, dates['test'], als_candidates_test)

    # -------------------------- Сохранение и отключение ------------------------- #
    # Сохранение датасета
    df_train.to_parquet(DATA_PROD_DIR / 'df_train.parquet', 
                        engine='pyarrow', index=False)
    df_valid.to_parquet(DATA_PROD_DIR / 'df_valid.parquet', 
                    engine='pyarrow', index=False)
    df_test.to_parquet(DATA_PROD_DIR / 'df_test.parquet', 
                       engine='pyarrow', index=False)

    # Отключение от БД
    con.close()

if __name__ == '__main__':
    run_feature_engineering()