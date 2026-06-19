# %%  Импорт библиотек
import pandas as pd

from catboost import CatBoostRanker, Pool
from scripts.map_at_k import map_at_k

# %% Импорт данных
df_train = pd.read_parquet('data/processed/df_train.parquet', engine='pyarrow')

# %% Правки
df_train = df_train.drop('article_first_purchase_date', axis=1)
df_train = df_train.sort_values('customer_id')

# %%
y_train = df_train['target']
X_train = df_train.drop(['customer_id', 'article_id', 'target'], axis=1)
i_train = df_train['customer_id']

train_pool = Pool(
    data=X_train,
    label=y_train,
    group_id=i_train
)
# %%
model = CatBoostRanker(
    loss_function='YetiRank',
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    random_seed=42,
    verbose=100
)

model.fit(train_pool)


# %%
y_hat_train = model.predict(train_pool)

# %%
map_at_k()
