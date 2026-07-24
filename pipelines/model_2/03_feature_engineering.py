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

# %% Характеристики выборок (название файла, временные интервалы)
splits = {
    'df_train.parquet': dates['train'],
    'df_valid.parquet': dates['valid'],
    'df_test.parquet': dates['test']
}

# %% Генерация выборок для LTR
k = 100 # кол-во генерируемых рекомендаций

for file, split_dates in splits.items():

    # Создание данных для энкодинга (покупатели, товары)
    candidates = generate_tt_inference(split_dates)

    # Two-tower: построение эмбеддингов + k лучших рекомендаций
    candidates = generate_tt_candidates(tt_model,
                                        candidates,
                                        k=k)

    # Генерация фичей для LTR + сохранение итоговой таблицы
    build_feature_table(split_dates, 
                        candidates, 
                        file,
                        DATA_PROCESSED_MOD2_DIR)
