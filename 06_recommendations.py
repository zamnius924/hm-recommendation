# %%  Импорт библиотек
import json
import pandas as pd

from catboost import CatBoostRanker
from scripts.generate_als_candidates import generate_als_candidates
from scripts.generate_features import generate_features
from scripts.generate_scores import generate_scores
from scripts.load_db import load_db
from scripts.paths import DATA_PROCESSED_DIR, DATA_RECOMMENDATIONS_DIR, MODELS_DIR, MODELS_CONFIG_DIR

# %% Development: Перезапуск 
import importlib
import scripts.generate_als_candidates
import scripts.generate_features
import scripts.generate_pool
import scripts.generate_scores

importlib.reload(scripts.generate_als_candidates)
importlib.reload(scripts.generate_features)
importlib.reload(scripts.generate_pool)
importlib.reload(scripts.generate_scores)

from scripts.generate_als_candidates import generate_als_candidates
from scripts.generate_features import generate_features
from scripts.generate_pool import generate_pool
from scripts.generate_scores import generate_scores

# %% Импорт
# Загрузка параметров: даты
with open(file=DATA_PROCESSED_DIR / 'split_dates.json', mode='r') as file:
    dates = json.load(file)

# Загрузка параметров: калибровка ALS
with open(file=MODELS_CONFIG_DIR / 'als_best_params.json', mode='r') as file:
    als_best_params = json.load(file)

# Загрузка модели
model = CatBoostRanker()
model.load_model(MODELS_DIR / 'ltr_model.cbm')

# %% Подключение к БД
con = load_db()

# %% Окно для данных
# Отсечка – дата, на которую строим рекомендации
cutoff_date = pd.to_datetime(dates['train']['feature_window_end'])
# Длина окна, на котором собираем историю
window_length = pd.to_timedelta(dates['window_length']['feature'])

# Границы окна
inference_dates = {
    'feature_window_end': cutoff_date,
    'feature_window_start': cutoff_date - window_length
}

# %% Прогноз ALS
als_candidates = generate_als_candidates(con,
                                         inference_dates,
                                         als_best_params,
                                         target=False)

# %% Создание фичей
con = generate_features(con,
                        inference_dates,
                        als_candidates,
                        target=False,
                        return_con=True)

# %% Прогноз скоров на основе LTR
df_rec = generate_scores(con, model, 5000)

# %% Отключение от БД
con.close()

# %% Сохранение рекомендаций
df_rec.to_parquet(DATA_RECOMMENDATIONS_DIR / 'df_rec.parquet',
                  engine='pyarrow', index=False)
