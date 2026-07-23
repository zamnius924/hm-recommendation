# %%  Импорт библиотек
import json
import torch

from scripts.data.load_db import load_db
from scripts.data.save_db import save_db
from scripts.features.generate_features import generate_features
from scripts.two_tower.generate_loader import generate_loader
from scripts.two_tower.generate_tt_candidates import generate_tt_candidates
from scripts.two_tower.generate_tt_datasets import generate_tt_inference
from scripts.utils.paths import DATA_PROCESSED_DIR, DATA_PROCESSED_MOD2_DIR, \
    MODELS_DIR

# %% Импорт
# Временное разделение на train-, valid- и test-выборки
with open(file=DATA_PROCESSED_DIR / 'split_dates.json', mode='r') as file:
    dates = json.load(file)

# Загрузка модели Two-tower
tt_model = torch.load(
    f=MODELS_DIR / 'tt_model.pt',
    weights_only=False
)

# %% Создание данных для энкодинга (покупатели, товары)
customer_dataset, article_dataset, mapping = generate_tt_inference(dates['train'])

# %% Инициализация даталоадеров для покупателей и товаров
data_loader_customer = generate_loader(customer_dataset, mode='eval')
data_loader_article = generate_loader(article_dataset, mode='eval')

# %% Генерация кандидатов на основе Two-tower
candidates_df = generate_tt_candidates(
    tt_model,
    data_loader_customer,
    data_loader_article,
    mapping,
    k=100
)

# %% Генерация фичей
# Инициализация БД
con = load_db()

# Генерация фичей
con = generate_features(con, 
                        dates['train'], 
                        candidates_df, 
                        return_con=True)

# %% Сохранение таблицы
save_db(con, 
        'df_train.parquet', 
        path_dir=DATA_PROCESSED_MOD2_DIR)

# %% Отключение от БД
con.close()
