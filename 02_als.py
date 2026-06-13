# %%  Импорт библиотек
import pyarrow
import numpy as np
import pandas as pd

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k
from scipy.sparse import csr_matrix
from scripts.sparse_interaction_matrix import sparse_interaction_matrix
from scripts.build_mapping import build_mapping

# %% Чтение данных
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

# %% Обучение модели ALS
als_model = AlternatingLeastSquares(
    factors=100,  # Number of latent factors
    iterations=15,  # Number of iterations to train
    regularization=0.01, # Strength of regularisation parameter
    alpha=1.0,  # Confidence weighting factor
    random_state=42 # For reproducibility
)

als_model.fit(train_interaction_matrix)

# %% Оценка качества с помощью Precision@K
prec_at_10 = precision_at_k(
    als_model,
    train_interaction_matrix,
    test_interaction_matrix,
    K=10,
    show_progress=True
)
print(f"Precision@10: {prec_at_10:.4f}")
