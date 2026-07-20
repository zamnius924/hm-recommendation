import datetime as dt
import json

from scripts.data.load_db import load_db
from scripts.data.load_window_config import load_window_config
from scripts.utils.paths import DATA_PROD_DIR, MODELS_PROD_CONFIG_DIR

def run_update_windows():

    # ---------------------------- Загрузка параметров --------------------------- #
    # Длины окон для формирования таргета и фичей
    window_length = load_window_config()

    # Подключение к БД и загрузка данных
    con = load_db()


    # -------------------------- Переоценка границ окон -------------------------- #
    # Максимальная дата в данных 
    max_date = con.execute("""
        SELECT MAX(t_dat)
        FROM transactions
    """).fetchone()[0]

    # Окончания периодов
    max_date_train_target = max_date
    max_date_train_feature = max_date_train_target - \
        (window_length['target'] + dt.timedelta(days=1))

    max_date_valid_target = max_date_train_feature
    max_date_valid_feature = max_date_valid_target - \
        (window_length['target'] + dt.timedelta(days=1))

    max_date_test_target = max_date_valid_feature
    max_date_test_feature = max_date_test_target - \
        (window_length['target'] + dt.timedelta(days=1))

    # Словарь с границами окна
    dates = {
        'train': {
            'target_window_end': max_date_train_target,
            'target_window_start': max_date_train_target - window_length['target'],
            'feature_window_end': max_date_train_feature,
            'feature_window_start': max_date_train_feature - window_length['feature']
        },
        'valid': {
            'target_window_end': max_date_valid_target,
            'target_window_start': max_date_valid_target - window_length['target'],
            'feature_window_end': max_date_valid_feature,
            'feature_window_start': max_date_valid_feature - window_length['feature']
        },
        'test': {
            'target_window_end': max_date_test_target,
            'target_window_start': max_date_test_target - window_length['target'],
            'feature_window_end': max_date_test_feature,
            'feature_window_start': max_date_test_feature - window_length['feature']
        },
        'window_length': window_length
    }


    # -------------------------- Сохранение и отключение ------------------------- #
    # Сохранение границ окон
    with open(file=DATA_PROD_DIR / 'split_dates.json', mode='w') as file:
        json.dump(dates, file, indent=4, default=str)

    # Отключение от БД
    con.close()


if __name__ == '__main__':
    run_update_windows()