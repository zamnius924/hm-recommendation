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

# Данные для Two-tower model
tt_data = generate_dataset(df_customer, df_article, df_pairs, mapping)

# %%
