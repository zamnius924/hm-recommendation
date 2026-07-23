# %%  Импорт библиотек
import json
import pandas as pd

from scripts.als.generate_als_candidates import generate_als_candidates
from scripts.features.generate_features import generate_features
from scripts.data.load_db import load_db
from scripts.utils.optimize_dtypes import optimize_dtypes
from scripts.utils.paths import DATA_PROCESSED_DIR, DATA_PROCESSED_MOD1_DIR, \
    MODELS_MOD1_CONFIG_DIR

# %% Development: Перезапуск 
import importlib
import scripts.als.generate_als_candidates
import scripts.features.generate_features

importlib.reload(scripts.als.generate_als_candidates)
importlib.reload(scripts.features.generate_features)

from scripts.als.generate_als_candidates import generate_als_candidates
from scripts.features.generate_features import generate_features

# %% Загрузка параметров
# Временное разделение на train-, valid- и test-выборки
with open(file=DATA_PROCESSED_DIR / 'split_dates.json', mode='r') as file:
    dates = json.load(file)

# Оптимальные параметры ALS
with open(file=MODELS_MOD1_CONFIG_DIR / 'als_best_params.json', mode='r') as file:
    als_best_params = json.load(file)

# %% Подключение к БД
con = load_db()

# %% Прогноз ALS
als_candidates_train = generate_als_candidates(con, dates['train'], als_best_params)
als_candidates_valid = generate_als_candidates(con, dates['valid'], als_best_params)
als_candidates_test = generate_als_candidates(con, dates['test'], als_best_params)

# %% Development: Сохранение кандидатов
optimize_dtypes(als_candidates_train).to_parquet(
    DATA_PROCESSED_MOD1_DIR / 'als_candidates_train.parquet',
    engine='pyarrow',
    index=False,
    compression='zstd'
)
optimize_dtypes(als_candidates_valid).to_parquet(
    DATA_PROCESSED_MOD1_DIR / 'als_candidates_valid.parquet',
    engine='pyarrow',
    index=False,
    compression='zstd'
)
optimize_dtypes(als_candidates_test).to_parquet(
    DATA_PROCESSED_MOD1_DIR / 'als_candidates_test.parquet',
    engine='pyarrow',
    index=False,
    compression='zstd'
)

# %% Development: Загрузка кандидатов
als_candidates_train = pd.read_parquet(
    DATA_PROCESSED_MOD1_DIR / 'als_candidates_train.parquet', 
    engine='pyarrow'
)
als_candidates_valid = pd.read_parquet(
    DATA_PROCESSED_MOD1_DIR / 'als_candidates_valid.parquet', 
    engine='pyarrow'
)
als_candidates_test = pd.read_parquet(
    DATA_PROCESSED_MOD1_DIR / 'als_candidates_test.parquet', 
    engine='pyarrow'
)

# %% Создание фичей
df_train = generate_features(con, dates['train'], als_candidates_train)
df_valid = generate_features(con, dates['valid'], als_candidates_valid)
df_test = generate_features(con, dates['test'], als_candidates_test)

# %% Отключение от БД
con.close()

# %% Сохранение датасета
optimize_dtypes(df_train).to_parquet(
    DATA_PROCESSED_MOD1_DIR / 'df_train.parquet',
    engine='pyarrow',
    index=False,
    compression='zstd'
)
optimize_dtypes(df_valid).to_parquet(
    DATA_PROCESSED_MOD1_DIR / 'df_valid.parquet',
    engine='pyarrow',
    index=False,
    compression='zstd'
)
optimize_dtypes(df_test).to_parquet(
    DATA_PROCESSED_MOD1_DIR / 'df_test.parquet',
    engine='pyarrow',
    index=False,
    compression='zstd'
)
