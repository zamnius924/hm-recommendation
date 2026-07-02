import json
import optuna
import pandas as pd

from scripts.generate_pool import generate_pool
from scripts.paths import DATA_PROD_DIR, MODELS_PROD_CONFIG_DIR
from scripts.tuning_objective_ltr import tuning_objective_ltr

def run_ltr_calibration():

    # ------------------------------ Загрузка данных ----------------------------- #
    # Датасеты
    df_train = pd.read_parquet(DATA_PROD_DIR / 'df_train.parquet', engine='pyarrow')
    df_valid = pd.read_parquet(DATA_PROD_DIR / 'df_valid.parquet', engine='pyarrow')


    # ---------------------- Калибровка гиперпараметров LTR ---------------------- #
    # Создание пулов
    pool_train = generate_pool(df_train)
    pool_valid = generate_pool(df_valid)

    # Фиксированные параметры для бустинга
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

    # Тюнинг гиперпараметров
    study = optuna.create_study(direction='maximize')

    study.optimize(
        lambda trial: tuning_objective_ltr(trial, 
                                           params,
                                           pool_train,
                                           pool_valid,
                                           df_valid), 
        n_trials=25
    )


    # -------------------------------- Сохранение -------------------------------- #
    # Сохранение оптимальных гиперпараметров
    ltr_best_params = {
        'model': params['model'] | study.best_params,
        'fit': params['fit']
    }

    with open(file=MODELS_PROD_CONFIG_DIR / 'ltr_best_params.json', mode='w') as file:
        json.dump(ltr_best_params, file, indent=4)


if __name__ == '__main__':
    run_ltr_calibration()