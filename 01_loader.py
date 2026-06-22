# %%  Импорт библиотек
import datetime as dt
import duckdb
import json
import pandas as pd

from scripts.load_db import load_db

# %% Подключение к БД и загрузка данных
con = load_db()
con.execute("SHOW TABLES").df()

# %% Даты начала и конца
df_dates = con.execute("""
    SELECT 
        MIN(t_dat) AS min_date,
        MAX(t_dat) AS max_date
    FROM transactions
""").df()

df_dates

# %% Определение границ тестовой, валидационной и тренировочной выборок
# Параметры окон
window_length_target = dt.timedelta(days=6)
window_length_feature = dt.timedelta(weeks=8)

# Окончания периодов
max_date_train_target = df_dates.max_date[0]
max_date_train_feature = max_date_train_target - \
    (window_length_target + dt.timedelta(days=1))

max_date_valid_target = max_date_train_feature
max_date_valid_feature = max_date_valid_target - \
    (window_length_target + dt.timedelta(days=1))

max_date_test_target = max_date_valid_feature
max_date_test_feature = max_date_test_target - \
    (window_length_target + dt.timedelta(days=1))

# Словарь с границами окон
dates = {
    'train': {
        'target_window_end': max_date_train_target,
        'target_window_start': max_date_train_target - window_length_target,
        'feature_window_end': max_date_train_feature,
        'feature_window_start': max_date_train_feature - window_length_feature
    },
    'valid': {
        'target_window_end': max_date_valid_target,
        'target_window_start': max_date_valid_target - window_length_target,
        'feature_window_end': max_date_valid_feature,
        'feature_window_start': max_date_valid_feature - window_length_feature
    },
    'test': {
        'target_window_end': max_date_test_target,
        'target_window_start': max_date_test_target - window_length_target,
        'feature_window_end': max_date_test_feature,
        'feature_window_start': max_date_test_feature - window_length_feature
    },
    'window_length': {
        'target': window_length_target,
        'feature': window_length_feature
    }
}

dates

# %% Сохранение граничиных дат
with open(file='data/processed/split_dates.json', mode='w') as file:
    json.dump(dates, file, indent=4, default=str)

# %% Отключение от БД
con.close()
