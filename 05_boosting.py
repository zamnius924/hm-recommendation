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
print(f'Most popular: MAP@12 (train sample) = {map_at_k(df_train, df_train.article_purchase_count_7d)}')
print(f'Most popular: MAP@12 (test sample) = {map_at_k(df_test, df_test.article_purchase_count_7d)}')

print(f'ALS: MAP@12 (train sample) = {map_at_k(df_train, df_train.article_score)}')
print(f'ALS: MAP@12 (test sample) = {map_at_k(df_test, df_test.article_score)}')

print(f'ALS + Boosting: MAP@12 (train sample) = {map_at_k(df_train, model.predict(pool_train))}')
print(f'ALS + Boosting: MAP@12 (test sample) = {map_at_k(df_test, model.predict(pool_test))}')

# %%
feature_importance_train = pd.DataFrame({
    'feature': pool_train.get_feature_names(),
    'importance': model.get_feature_importance(pool_train)
}).sort_values('importance', ascending=False)

feature_importance_test = pd.DataFrame({
    'feature': pool_train.get_feature_names(),
    'importance': model.get_feature_importance(pool_test)
}).sort_values('importance', ascending=False)

print(feature_importance_train.head(20))
print(feature_importance_test.head(20))

# %%
model.plot_tree(tree_idx=0)