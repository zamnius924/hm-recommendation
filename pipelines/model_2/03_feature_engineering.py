# %%  Импорт библиотек
import json
import torch

from scripts.features.build_feature_table import build_feature_table
from scripts.two_tower.generate_tt_candidates import generate_tt_candidates
from scripts.two_tower.generate_tt_datasets import generate_tt_inference
from scripts.utils.paths import DATA_PROCESSED_DIR, DATA_PROCESSED_MOD2_DIR, \
    MODELS_MOD2_DIR

# %% Импорт
# Временное разделение на train-, valid- и test-выборки
with open(file=DATA_PROCESSED_DIR / 'split_dates.json', mode='r') as file:
    dates = json.load(file)

# Загрузка модели Two-tower
tt_model = torch.load(
    f=MODELS_MOD2_DIR / 'tt_model.pt',
    weights_only=False
)

# %% Создание данных для энкодинга (покупатели, товары)
candidates_train = generate_tt_inference(dates['train'])
candidates_valid = generate_tt_inference(dates['valid'])
candidates_test = generate_tt_inference(dates['test'])

# %% Генерация кандидатов на основе Two-tower
k = 100 # кол-во генерируемых рекомендаций

candidates_train = generate_tt_candidates(
    tt_model,
    candidates_train,
    k=k
)
candidates_valid = generate_tt_candidates(
    tt_model,
    candidates_valid,
    k=k
)
candidates_test = generate_tt_candidates(
    tt_model,
    candidates_test,
    k=k
)

# %% Генерация и сохранение фичей
build_feature_table(dates['train'], 
                    candidates_train, 
                    'df_train.parquet',
                    DATA_PROCESSED_MOD2_DIR)
build_feature_table(dates['valid'], 
                    candidates_valid, 
                    'df_valid.parquet',
                    DATA_PROCESSED_MOD2_DIR)
build_feature_table(dates['test'],
                    candidates_test,
                    'df_test.parquet',
                    DATA_PROCESSED_MOD2_DIR)
