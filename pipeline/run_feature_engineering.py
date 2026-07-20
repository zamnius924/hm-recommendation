import argparse
import json

from scripts.als.generate_als_candidates import generate_als_candidates
from scripts.features.generate_features import generate_features
from scripts.data.load_db import load_db
from scripts.utils.paths import DATA_PROD_DIR, MODELS_PROD_CONFIG_DIR


def build_dataset(con, dates: dict, sample: str, als_best_params: dict):
    
    # Кандидаты ALS
    als_candidates = generate_als_candidates(con, dates[sample], als_best_params)

    # Создание фичей
    df = generate_features(con, dates[sample], als_candidates)

    return df


def save_dataset(df, sample: str):
    
    # Сохранение датасета
    df.to_parquet(DATA_PROD_DIR / f'df_{sample}.parquet', 
                  engine='pyarrow', index=False)
    

def run_feature_engineering(mode: str):

    # ---------------------------- Загрузка параметров --------------------------- #
    # Границы окон
    with open(file=DATA_PROD_DIR / 'split_dates.json', mode='r') as file:
        dates = json.load(file)

    # Оптимальные параметры ALS
    with open(MODELS_PROD_CONFIG_DIR / 'als_best_params.json', mode='r') as file:
        als_best_params = json.load(file)

    # Покдючение к БД
    con = load_db()


    # --------------------- Обработка и сохранение датасетов --------------------- #
    if mode == 'production': # Обучение модели на новых данных (без калибровки гиперпараметров)

        # Создание датасетов
        df_train = build_dataset(con, dates, 'train', als_best_params)
        df_test = build_dataset(con, dates, 'test', als_best_params)
        
        # Сохранение датасетов
        save_dataset(df_train, 'train')
        save_dataset(df_test, 'test')

    elif mode == 'calibration': # Калибровка гиперпараметров

        # Создание датасетов
        df_train = build_dataset(con, dates, 'train', als_best_params)
        df_valid = build_dataset(con, dates, 'valid', als_best_params)

        # Сохранение датасетов
        save_dataset(df_train, 'train')
        save_dataset(df_valid, 'valid')
    
    else:

        raise ValueError(f'Unknown mode: {mode}')

    # Отключение от БД
    con.close()


if __name__ == '__main__':

    # Инициализация парсера аргументов командной строки
    parser = argparse.ArgumentParser()

    # Описание переменной mode
    parser.add_argument('--mode', type=str, choices=['production', 'calibration'])

    # Сохранение аргументов командной строки
    args = parser.parse_args()

    # Вызов функции обработки данных с аргументами командной строки
    run_feature_engineering(mode=args.mode)