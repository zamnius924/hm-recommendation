# %%  Импорт библиотек
import pyarrow
import numpy as np
import pandas as pd

from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from scripts.sparse_interaction_matrix import sparse_interaction_matrix

# %% Чтение данных
df_train = pd.read_parquet('data/processed/dataset_train.parquet', engine='pyarrow')
df_test = pd.read_parquet('data/processed/dataset_test.parquet', engine='pyarrow')

# %% Матрица взаимодействия
(
    train_interaction_matrix,
    customer_id_map,
    article_id_map,
    customer_index_map,
    article_index_map
) = sparse_interaction_matrix(df_train)

# %%
als_model = AlternatingLeastSquares(
    factors=100,  # Number of latent factors
    iterations=15,  # Number of iterations to train
    regularization=0.01, # Strength of regularisation parameter
    alpha=1.0,  # Confidence weighting factor
    random_state=42 # For reproducibility
)

als_model.fit(train_interaction_matrix)
