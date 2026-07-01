# %%  Импорт библиотек
import json
import optuna

from scripts.tuning_objective_als import tuning_objective_als
from scripts.build_mapping import build_mapping
from scripts.load_db import load_db
from scripts.paths import DATA_PROCESSED_DIR, MODELS_CONFIG_DIR
from scripts.sparse_interaction_matrix import sparse_interaction_matrix
from scripts.window_extraction import window_extraction

# %% Подключение к БД и загрузка данных
con = load_db()
con.execute("SHOW TABLES").df()

# Временное разделение на train-, valid- и test-выборки
with open(file=DATA_PROCESSED_DIR / 'split_dates.json', mode='r') as file:
    dates = json.load(file)

# %% Выделение target- и feature-window на valid-выборке
df_feature, df_target = window_extraction(con, dates['valid'])

# %% Создание мэппинга для feature-window
mapping = build_mapping(df_feature)

# %% Оставляем на target-window только тех, кто был в feature-window
df_target = df_target[
    df_target.customer_id.isin(mapping["customer_id2index"])
    & df_target.article_id.isin(mapping["article_id2index"])
]

print(f'Кол-во наблюдений на test после фильтрации: {len(df_target)}')

# %% Матрицы взаимодействия
# feature-window
feature_interaction_matrix = sparse_interaction_matrix(
    df_feature,
    mapping['customer_id2index'],
    mapping['article_id2index']
)
# target-window
target_interaction_matrix = sparse_interaction_matrix(
    df_target,
    mapping['customer_id2index'],
    mapping['article_id2index']
)

# %% Количество рекомендаций на каждого пользователя
n_recommendation = 100

# %% Тюнинг гиперпараметров
study = optuna.create_study(direction='maximize')

study.optimize(
    lambda trial: tuning_objective_als(trial, 
                                       feature_interaction_matrix,
                                       target_interaction_matrix,
                                       n_recommendation), 
    n_trials=25
)

print(f"\nBest params: {study.best_params}")
print(f"Best MAP@{n_recommendation}: {study.best_value:.4f}")

# %% Сохранение оптимальных гиперпараметров
als_best_params = study.best_params
als_best_params['K'] = n_recommendation

with open(file=MODELS_CONFIG_DIR / 'als_best_params.json', mode='w') as file:
    json.dump(als_best_params, file, indent=4)

# %% Отключение от БД
con.close()
