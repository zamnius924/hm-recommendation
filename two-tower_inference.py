# %%  Импорт библиотек
import json
import torch

from scripts.two_tower.generate_loader import generate_loader
from scripts.two_tower.generate_tt_datasets import generate_tt_inference
from scripts.two_tower.tt_model_candidates import tt_model_candidates
from scripts.utils.paths import DATA_PROCESSED_DIR, MODELS_DIR

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

# %% Инициализация даталоадеров для покупателей и товаров
data_loader_customer = generate_loader(customer_dataset, mode='eval')
data_loader_article = generate_loader(article_dataset, mode='eval')

# %% Генерация кандидатов
tt_candidates_idx, tt_candidates_val = tt_model_candidates(
    tt_model,
    data_loader_customer,
    data_loader_article,
    k=100
)
# %%
