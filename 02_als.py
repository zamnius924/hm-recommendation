# %%  Импорт библиотек
import json
import numpy as np
import optuna
import pandas as pd
import pyarrow

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k
from scipy.sparse import csr_matrix
from scripts.als_tuning_objective import als_tuning_objective
from scripts.build_mapping import build_mapping
from scripts.sparse_interaction_matrix import sparse_interaction_matrix

# %% Импорт данных
df_train = pd.read_parquet('data/processed/dataset_train.parquet', engine='pyarrow')
df_test = pd.read_parquet('data/processed/dataset_test.parquet', engine='pyarrow')

print(f'Кол-во наблюдений на train: {len(df_train)}')
print(f'Кол-во наблюдений на test: {len(df_test)}')

# %% Создание мэппинга для train-выборки
mapping = build_mapping(df_train)

# %% Остсавляем на test-выборке только тех, кто был в train-выборке (train-only mapping)
df_test = df_test[
    df_test.customer_id.isin(mapping["customer_id2index"])
    & df_test.article_id.isin(mapping["article_id2index"])
]

print(f'Кол-во наблюдений на test после фильтрации: {len(df_test)}')

# %% Матрицы взаимодействия
# Train-выборка
train_interaction_matrix = sparse_interaction_matrix(
    df_train,
    mapping['customer_id2index'],
    mapping['article_id2index']
)
# Test-выборка
test_interaction_matrix = sparse_interaction_matrix(
    df_test,
    mapping['customer_id2index'],
    mapping['article_id2index']
)

# %% Тюнинг гиперпараметров
study = optuna.create_study(direction='maximize')

study.optimize(
    lambda trial: als_tuning_objective(trial, train_interaction_matrix, test_interaction_matrix), 
    n_trials=25
)

print(f"\nBest params: {study.best_params}")
print(f"Best precision@10: {study.best_value:.4f}")

# %% Сохранение оптимальных гиперпараметров
with open(file='models/als_best_params.json', mode='w') as file:
    json.dump(study.best_params, file, indent=4)
