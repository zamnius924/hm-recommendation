# %%  Импорт библиотек
import pandas as pd

from catboost import CatBoostRanker
from scripts.generate_pool import generate_pool
from scripts.map_at_k import map_at_k

# %% Импорт данных
df_train = pd.read_parquet('data/processed/df_train.parquet', engine='pyarrow')
df_test = pd.read_parquet('data/processed/df_test.parquet', engine='pyarrow')

# %% Создание пулов
pool_train = generate_pool(df_train)
pool_test = generate_pool(df_test)

# %% Обучение CatBoost
model = CatBoostRanker(
    loss_function='YetiRank',
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    random_seed=42,
    verbose=100
)

model.fit(pool_train)

# %% Оценка качества
print(f'MAP@12 (train sample) = {map_at_k(df_train, model.predict(pool_train))}')
print(f'MAP@12 (test sample) = {map_at_k(df_test, model.predict(pool_test))}')
