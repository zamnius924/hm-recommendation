# %%  Импорт библиотек
import json
import numpy as np
import torch
import torch.nn as nn

from scripts.aggregate_tt_dfs import aggregate_tt_dfs
from scripts.build_mapping import build_mapping
from scripts.class_loss import SymmetricCrossEntropyLoss
from scripts.class_towers import TwoTower
from scripts.class_tt_data import TwoTowerDataset, TowerInfo
from scripts.generate_tt_dfs import generate_tt_dfs
from scripts.load_db import load_db
from scripts.paths import DATA_PROCESSED_DIR, MODELS_CONFIG_DIR
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# %% Загрузка параметров
# Временное разделение на train-, valid- и test-выборки
with open(file=DATA_PROCESSED_DIR / 'split_dates.json', mode='r') as file:
    dates = json.load(file)

# %% Создание датафреймов с фичами и индексами пар
# Подключение к БД
con = load_db()

# Датафреймы с фичами и индексами пар
df_customer, df_article, df_pairs = generate_tt_dfs(con, 
                                                    dates['train'], 
                                                    return_df=True)

# Отключение от БД
con.close()

# %% Форматирование данных для Two-tower model
# Создание мэппинга
mapping = build_mapping(df_customer, df_article, df_pairs)

# Вспомогательные данные
df_aggr = aggregate_tt_dfs(df_customer, df_article, df_pairs, mapping)

# Данные для Two-tower model
tt_info = TowerInfo(df_aggr)
tt_dataset = TwoTowerDataset(df_aggr)

# %% Инициализация модели
# Экземпляр сети
tt_model = TwoTower(
    num_hidden_dim_customer=64,
    num_hidden_dim_article=64,
    emb_dims_customer=[8, 8], 
    emb_dims_article=[],
    tt_info=tt_info
)

# Функция потерь
criterion = SymmetricCrossEntropyLoss()

# Даталоадер
loader = DataLoader(
    tt_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=0
)

# Оптимизатор
optimizer = optim.Adam(tt_model.parameters(), lr=1e-3)

# %%
# Перевод модели в режим обучения
tt_model.train()

for batch in tqdm(loader):

    # Обнуление градиента
    optimizer.zero_grad()

    # Эмбеддинги покупателей и товаров
    u, v = tt_model(
        batch['customer_num'], 
        batch['customer_cat'], 
        batch['article_num'], 
        batch['article_cat']
    )

    # Матрица скалярных произведений
    logits = tt_model.similarity(u, v)

    # Значение функционала потерь
    loss = criterion(logits)

    # backward pass – оценка производных и градиента функции потерь
    loss.backward()

    # Шаг градиентного спуска – новые значения параметров
    optimizer.step()


# %%
batch_size = 10

x_num_customer = tt_dataset[0:batch_size]['customer_num']
x_cat_customer = tt_dataset[0:batch_size]['customer_cat']

x_num_article = tt_dataset[0:batch_size]['article_num']
x_cat_article = tt_dataset[0:batch_size]['article_cat']

# %%
u, v = tt_model(
    x_num_customer, 
    x_cat_customer, 
    x_num_article, 
    x_cat_article
)

# %% Матрица скалярных произведений
logits = tt_model.similarity(u, v)

#target = torch.arange(logits.shape[0], device=logits.device)

# %%
loss = criterion(logits)

# %%
batch = next(iter(loader))






# %%
