import json
import optuna

from scripts.tuning_objective_als import tuning_objective_als
from scripts.build_mapping import build_mapping
from scripts.load_db import load_db
from scripts.paths import DATA_PROD_DIR, MODELS_PROD_CONFIG_DIR
from scripts.sparse_interaction_matrix import sparse_interaction_matrix
from scripts.window_extraction import window_extraction


def run_als_calibration():

    # ---------------------------- Загрузка параметров --------------------------- #
    # Границы окон
    with open(file=DATA_PROD_DIR / 'split_dates.json', mode='r') as file:
        dates = json.load(file)

    # Покдючение к БД
    con = load_db()


    # ----------------------------- Обработка данных ----------------------------- #
    # Выделение target- и feature-window на valid-выборке
    df_feature, df_target = window_extraction(con, dates['valid'])

    # Создание мэппинга для feature-window
    mapping = build_mapping(df_feature)

    # Оставляем на target-window только тех, кто был в feature-window
    df_target = df_target[
        df_target.customer_id.isin(mapping["customer_id2index"])
        & df_target.article_id.isin(mapping["article_id2index"])
    ]

    # Матрица взаимодействия: feature-window
    feature_interaction_matrix = sparse_interaction_matrix(
        df_feature,
        mapping['customer_id2index'],
        mapping['article_id2index']
    )
    # Матрица взаимодействия: target-window
    target_interaction_matrix = sparse_interaction_matrix(
        df_target,
        mapping['customer_id2index'],
        mapping['article_id2index']
    )

    
    # ---------------------- Калибровка гиперпараметров ALS ---------------------- #
    # Количество рекомендаций на каждого пользователя
    n_recommendation = 100

    # Тюнинг гиперпараметров
    study = optuna.create_study(direction='maximize')

    study.optimize(
        lambda trial: tuning_objective_als(trial, 
                                           feature_interaction_matrix,
                                           target_interaction_matrix,
                                           n_recommendation), 
        n_trials=25
    )


    # -------------------------- Сохранение и отключение ------------------------- #
    # Сохранение оптимальных гиперпараметров
    als_best_params = study.best_params
    als_best_params['K'] = n_recommendation

    with open(file=MODELS_PROD_CONFIG_DIR / 'als_best_params.json', mode='w') as file:
        json.dump(als_best_params, file, indent=4)

    # Отключение от БД
    con.close()


if __name__ == '__main__':
    run_als_calibration()