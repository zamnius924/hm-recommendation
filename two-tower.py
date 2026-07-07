# %%  Импорт библиотек
import json
import numpy as np
import torch
import torch.nn as nn

from scripts.build_mapping import build_mapping
from scripts.generate_dataset import generate_dataset
from scripts.generate_tt_data import generate_tt_data
from scripts.load_db import load_db
from scripts.paths import DATA_PROCESSED_DIR, MODELS_CONFIG_DIR
from scripts.class_towers import Tower, TwoTower
from scripts.class_tt_data import TwoTowerDataset

# %% Загрузка параметров
# Временное разделение на train-, valid- и test-выборки
with open(file=DATA_PROCESSED_DIR / 'split_dates.json', mode='r') as file:
    dates = json.load(file)

# %% Создание датафреймов с фичами и индексами пар
# Подключение к БД
con = load_db()

# Датафреймы с фичами и индексами пар
df_customer, df_article, df_pairs = generate_tt_data(con, 
                                                     dates['train'], 
                                                     return_df=True)

# Отключение от БД
con.close()

# %% Форматирование данных для Two-tower model
# Создание мэппинга
mapping = build_mapping(df_customer, df_article, df_pairs)

# Вспомогательные данные
df_aux = generate_dataset(df_customer, df_article, df_pairs, mapping)

# Данные для Two-tower model
tt_dataset = TwoTowerDataset(df_aux)

# %%
customer_tower = Tower(
    num_input_dim=tt_dataset.customer_num_dim,
    num_hidden_dim=64,
    cat_sizes=tt_dataset.customer_cat_sizes,
    emb_dims=[8, 8]
)

article_tower = Tower(
    num_input_dim=tt_dataset.article_num_dim,
    num_hidden_dim=64,
    cat_sizes=tt_dataset.article_cat_sizes,
    emb_dims=[]
)

# %%
batch_size = 10

x_num = tt_dataset[0:batch_size]['article_num']
x_cat = tt_dataset[0:batch_size]['article_cat']

# %%
#customer_tower(x_num, x_cat)
article_tower(x_num, x_cat)

# %%
#train_loader = DataLoader(
#    train_dataset,
#    batch_size=1024,
#    shuffle=True,
#    drop_last=True,  # stable batch shapes
#    num_workers=0,   # use 0 for notebook simplicity; increase for speed
#)