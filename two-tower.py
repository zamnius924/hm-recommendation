# %%  Импорт библиотек
import json
import numpy as np
import torch
import torch.nn as nn

from scripts.aggregate_tt_dfs import aggregate_tt_dfs
from scripts.build_mapping import build_mapping
from scripts.class_towers import Tower, TwoTower
from scripts.class_tt_data import TwoTowerDataset, TowerInfo
from scripts.generate_tt_dfs import generate_tt_dfs
from scripts.load_db import load_db
from scripts.paths import DATA_PROCESSED_DIR, MODELS_CONFIG_DIR

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

# %%
tt_model = TwoTower(
    num_hidden_dim_customer=64,
    num_hidden_dim_article=64,
    emb_dims_customer=[8, 8], 
    emb_dims_article=[],
    tt_info=tt_info
)

# %%
customer_tower = Tower(
    num_input_dim=tt_info.customer_num_dim,
    num_hidden_dim=64,
    cat_sizes=tt_info.customer_cat_sizes,
    emb_dims=[8, 8]
)

article_tower = Tower(
    num_input_dim=tt_info.article_num_dim,
    num_hidden_dim=64,
    cat_sizes=tt_info.article_cat_sizes,
    emb_dims=[]
)

# %%
batch_size = 10

x_num_customer = tt_dataset[0:batch_size]['customer_num']
x_cat_customer = tt_dataset[0:batch_size]['customer_cat']

x_num_article = tt_dataset[0:batch_size]['article_num']
x_cat_article = tt_dataset[0:batch_size]['article_cat']

# %%
y_customer = customer_tower(x_num_customer, x_cat_customer)
y_article = article_tower(x_num_article, x_cat_article)

# %%
y_customer @ y_article.T

# %%
#train_loader = DataLoader(
#    train_dataset,
#    batch_size=1024,
#    shuffle=True,
#    drop_last=True,  # stable batch shapes
#    num_workers=0,   # use 0 for notebook simplicity; increase for speed
#)