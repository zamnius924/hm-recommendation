# %%  Импорт библиотек
import json
import torch

from scripts.generate_tt_inference import generate_tt_inference
from scripts.paths import DATA_PROCESSED_DIR, MODELS_DIR

# %% Загрузка
# Временное разделение на train-, valid- и test-выборки
with open(file=DATA_PROCESSED_DIR / 'split_dates.json', mode='r') as file:
    dates = json.load(file)

# Two-tower
tt_model = torch.load(
    f=MODELS_DIR / 'tt_model.pt',
    weights_only=False
)
# %%
customer_dataset, article_dataset, mapping = generate_tt_inference(dates['train'])

# %%
