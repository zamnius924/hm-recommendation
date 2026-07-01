# %%  Импорт библиотек
import json
import optuna
import pandas as pd

from scripts.tuning_objective_ltr import tuning_objective_ltr
from scripts.generate_pool import generate_pool

# %% Импорт данных
df_train = pd.read_parquet('data/processed/df_train.parquet', engine='pyarrow')
df_valid = pd.read_parquet('data/processed/df_valid.parquet', engine='pyarrow')

# %% Создание пулов
pool_train = generate_pool(df_train)
pool_valid = generate_pool(df_valid)

# %% Фиксированные параметры для бустинга
params = {
    'model': {
        'loss_function': 'YetiRank',
        'iterations': 5000,
        'random_seed': 42,
    },
    'fit': {
        'use_best_model': True,
        'early_stopping_rounds': 200,
        'verbose': False
    }
}

# %% Тюнинг гиперпараметров
study = optuna.create_study(direction='maximize')

study.optimize(
    lambda trial: tuning_objective_ltr(trial, 
                                       params,
                                       pool_train,
                                       pool_valid,
                                       df_valid), 
    n_trials=25
)

print(f"\nBest params: {study.best_params}")
print(f"Best MAP@12: {study.best_value:.4f}")

# %% Сохранение оптимальных гиперпараметров
ltr_best_params = {
    'model': params['model'] | study.best_params,
    'fit': params['fit']
}

with open(file='models/config/ltr_best_params.json', mode='w') as file:
    json.dump(ltr_best_params, file, indent=4)
