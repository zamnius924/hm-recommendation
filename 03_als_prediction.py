# %%  Импорт библиотек
import json
import numpy as np
import pandas as pd
import pyarrow

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k
from scipy.sparse import csr_matrix
from scripts.build_mapping import build_mapping
from scripts.sparse_interaction_matrix import sparse_interaction_matrix

# %% Импорт
# Train-выборка
df_train = pd.read_parquet('data/processed/dataset_train.parquet', engine='pyarrow')
print(f'Кол-во наблюдений на train: {len(df_train)}')

# Оптимальные параметры ALS
with open('models/als_best_params.json', mode='r') as file:
    als_best_params = json.load(file)

# %% Количество рекомендаций на каждого пользователя
n_recommendation = 100

# %% Создание мэппинга и матрицы взимодействия
mapping = build_mapping(df_train)
train_interaction_matrix = sparse_interaction_matrix(
    df_train,
    mapping['customer_id2index'],
    mapping['article_id2index']
)

# %% Обучение модели ALS
als_model = AlternatingLeastSquares(
    factors=als_best_params['factors'], # Размеры эмбеддингов
    iterations=als_best_params['iterations'], # Кол-во итераций при обучении
    regularization=als_best_params['regularization'], # Параметр регуляризации
    alpha=als_best_params['alpha'], # Вес сигнала
    random_state=42
)

als_model.fit(train_interaction_matrix)

# %% Генерация кандидатов на train-выборке
# Индексы покупателей, для которых построены рекомендации
customer_index = [
    mapping['customer_id2index'][customer]
    for customer in df_train.customer_id.unique()
]

# Индексы и оценки рекомендуемых товаров
article_index, article_score = als_model.recommend(
    userid=customer_index,
    user_items=train_interaction_matrix,
    N=n_recommendation
)

# %% Таблциа с рекомендациями
# Таблица с индексами покупателей и товаров
df_recom = pd.DataFrame({
    'customer_id': np.repeat(customer_index, n_recommendation),
    'article_id': article_index.ravel(),
    'article_score': article_score.ravel()
})

# Перевод индексов покупателей и товаров в идентификаторы
df_recom.customer_id = df_recom.customer_id.map(mapping['customer_index2id'])
df_recom.article_id = df_recom.article_id.map(mapping['article_index2id'])

# %% Сохранение результата
df_recom.to_parquet('data/processed/als_candidates.parquet', 
                    engine='pyarrow', index=False)



