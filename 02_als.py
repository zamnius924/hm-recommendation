# %%  Импорт библиотек
import pyarrow
import numpy as np
import pandas as pd

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k
from scipy.sparse import csr_matrix
from scripts.sparse_interaction_matrix import sparse_interaction_matrix

# %% Чтение данных
df_train = pd.read_parquet('data/processed/dataset_train.parquet', engine='pyarrow')
df_test = pd.read_parquet('data/processed/dataset_test.parquet', engine='pyarrow')

# %% Матрица взаимодействия
result_train = sparse_interaction_matrix(df_train)
result_test = sparse_interaction_matrix(df_test)

train_interaction_matrix = result_train['matrix']
test_interaction_matrix = result_test['matrix']

del result_train, result_test

# %% Обучение модели ALS
als_model = AlternatingLeastSquares(
    factors=100,  # Number of latent factors
    iterations=15,  # Number of iterations to train
    regularization=0.01, # Strength of regularisation parameter
    alpha=1.0,  # Confidence weighting factor
    random_state=42 # For reproducibility
)

als_model.fit(train_interaction_matrix)

# %% ОЦенка качества с помощью Precision@K
prec_at_10 = precision_at_k(
    als_model,
    train_interaction_matrix,
    test_interaction_matrix,
    K=10,
    show_progress=True
)
print(f"Precision@10: {prec_at_10:.4f}")