# %%  Импорт библиотек
import json
import pandas as pd

from scripts.generate_als_candidates import generate_als_candidates
from scripts.generate_features import generate_features
from scripts.load_db import load_db
from scripts.paths import DATA_PROCESSED_DIR, MODELS_CONFIG_DIR

# %% Development: Перезапуск 
import importlib
import scripts.generate_als_candidates
import scripts.generate_features

importlib.reload(scripts.generate_als_candidates)
importlib.reload(scripts.generate_features)

from scripts.generate_als_candidates import generate_als_candidates
from scripts.generate_features import generate_features

# %% Загрузка параметров
# Временное разделение на train-, valid- и test-выборки
with open(file=DATA_PROCESSED_DIR / 'split_dates.json', mode='r') as file:
    dates = json.load(file)

# Оптимальные параметры ALS
with open(file=MODELS_CONFIG_DIR / 'als_best_params.json', mode='r') as file:
    als_best_params = json.load(file)

# %% Подключение к БД
con = load_db()

# %% Прогноз ALS
als_candidates_train = generate_als_candidates(con, dates['train'], als_best_params)
als_candidates_valid = generate_als_candidates(con, dates['valid'], als_best_params)
als_candidates_test = generate_als_candidates(con, dates['test'], als_best_params)

# %% Development: Сохранение кандидатов
als_candidates_train.to_parquet(DATA_PROCESSED_DIR / 'als_candidates_train.parquet', 
                                engine='pyarrow', index=False)
als_candidates_valid.to_parquet(DATA_PROCESSED_DIR / 'als_candidates_valid.parquet', 
                                engine='pyarrow', index=False)
als_candidates_test.to_parquet(DATA_PROCESSED_DIR / 'als_candidates_test.parquet', 
                               engine='pyarrow', index=False)

# %% Development: Загрузка кандидатов
als_candidates_train = pd.read_parquet(DATA_PROCESSED_DIR / 'als_candidates_train.parquet', 
                                       engine='pyarrow')
als_candidates_valid = pd.read_parquet(DATA_PROCESSED_DIR / 'als_candidates_valid.parquet', 
                                       engine='pyarrow')
als_candidates_test = pd.read_parquet(DATA_PROCESSED_DIR / 'als_candidates_test.parquet', 
                                      engine='pyarrow')

# %% Создание фичей
df_train = generate_features(con, dates['train'], als_candidates_train)
df_valid = generate_features(con, dates['valid'], als_candidates_valid)
df_test = generate_features(con, dates['test'], als_candidates_test)

# %% Отключение от БД
con.close()

# %% Сохранение датасета
df_train.to_parquet(DATA_PROCESSED_DIR / 'df_train.parquet', 
                    engine='pyarrow', index=False)
df_valid.to_parquet(DATA_PROCESSED_DIR / 'df_valid.parquet', 
                    engine='pyarrow', index=False)
df_test.to_parquet(DATA_PROCESSED_DIR / 'df_test.parquet', 
                   engine='pyarrow', index=False)
