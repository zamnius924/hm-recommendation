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
# Окончания периодов
max_date_train = df_dates.max_date[0]
max_date_valid = df_dates.max_date[0] - dt.timedelta(weeks=1)
max_date_test = df_dates.max_date[0] - dt.timedelta(weeks=2)

# Словарь с границами окон
dates = {
    'train':{
        'target_window_end': max_date_train,
        'target_window_start': max_date_train - dt.timedelta(days=6),
        'feature_window_end': max_date_train - dt.timedelta(weeks=1),
        'feature_window_start': max_date_train - dt.timedelta(weeks=9)
    },
    'valid':{
        'target_window_end': max_date_valid,
        'target_window_start': max_date_valid - dt.timedelta(days=6),
        'feature_window_end': max_date_valid - dt.timedelta(weeks=1),
        'feature_window_start': max_date_valid - dt.timedelta(weeks=9)
    },
    'test':{
        'target_window_end': max_date_test,
        'target_window_start': max_date_test - dt.timedelta(days=6),
        'feature_window_end': max_date_test - dt.timedelta(weeks=1),
        'feature_window_start': max_date_test - dt.timedelta(weeks=9)
    }
}

dates

# %% Сохранение граничиных дат
with open(file='data/processed/split_dates.json', mode='w') as file:
    json.dump(dates, file, indent=4, default=str)

# Закрытие подключение к БД
con.close()
