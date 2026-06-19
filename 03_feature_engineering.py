# %%  Импорт библиотек
import datetime as dt
import duckdb
import json
import pandas as pd

from scripts.generate_als_candidates import generate_als_candidates
from scripts.generate_features import generate_features
from scripts.load_db import load_db

# %% 
import importlib
import scripts.generate_als_candidates
import scripts.generate_features

importlib.reload(scripts.generate_als_candidates)
importlib.reload(scripts.generate_features)

from scripts.generate_als_candidates import generate_als_candidates
from scripts.generate_features import generate_features

# %% Загрузка параметров
# Временное разделение на train-, valid- и test-выборки
with open(file='data/processed/split_dates.json', mode='r') as file:
    dates = json.load(file)

# Оптимальные параметры ALS
with open('models/als_best_params.json', mode='r') as file:
    als_best_params = json.load(file)

# %% Подключение к БД
con = load_db()

# %% Прогноз ALS
als_candidates = generate_als_candidates(con, dates['train'], als_best_params)

# %% Создание фичей
df_train = generate_features(con, dates['train'], als_candidates)

# %% Отключение от БД
con.close()

# %% Сохранение датасета
df_train.to_parquet('data/processed/df_train.parquet', 
                    engine='pyarrow', index=False)
