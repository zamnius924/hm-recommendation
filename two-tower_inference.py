# %%  Импорт библиотек
import json
import torch

from scripts.generate_tt_inference import generate_tt_inference
from scripts.paths import DATA_PROCESSED_DIR, MODELS_DIR

from scripts.tt_model_candidates import tt_model_candidates

# %% Загрузка
# Временное разделение на train-, valid- и test-выборки
with open(file=DATA_PROCESSED_DIR / 'split_dates.json', mode='r') as file:
    dates = json.load(file)

# Two-tower
tt_model = torch.load(
    f=MODELS_DIR / 'tt_model.pt',
    weights_only=False
)

# %% Создание данных для энкодинга (покупатели, товары)
customer_dataset, article_dataset, mapping = generate_tt_inference(dates['train'])

# %% Генерация кандидатов
tt_candidates_idx, tt_candidates_val = tt_model_candidates(
    tt_model,
    customer_dataset,
    article_dataset,
    k=100
)
# %%
