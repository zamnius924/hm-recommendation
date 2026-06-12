# %%  Импорт библиотек
import duckdb
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

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

# %% Определение границ тестовой и тренировочной выборок
dates = {
    'test_end':     df_dates.max_date[0],
    'test_start':   df_dates.max_date[0] - dt.timedelta(days=6),
    'train_end':    df_dates.max_date[0] - dt.timedelta(weeks=1),
    'train_start':  df_dates.max_date[0] - dt.timedelta(weeks=9)
}

dates
