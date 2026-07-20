import optuna

from catboost import CatBoostRanker
from scripts.evaluation.map_at_k import map_at_k

def tuning_objective_ltr(trial, params, pool_train, pool_valid, df_valid):

    # Предопределение
    model_params = params['model'].copy()

    # Возможные значения гиперпараметров
    model_params['depth'] = trial.suggest_int( # глубина деревьев
        'depth', 4, 12, step=1
    ) 
    #iterations = trial.suggest_int( # кол-во деревьев
    #    'iterations', 300, 1500, step=100
    #) 
    model_params['learning_rate'] = trial.suggest_float( # шаг в градиентном бустинге
        'learning_rate', 0.01, 0.2, log=True
    ) 
    model_params['l2_leaf_reg'] = trial.suggest_float( # регуляризация на листьях
        'l2_leaf_reg', 1, 100, log=True
    ) 

    # Обучение модели LTR
    model = CatBoostRanker(**model_params)

    model.fit(
        pool_train,
        eval_set=pool_valid,
        **params['fit']
    )

    # Оценка качества
    quality = map_at_k(df_valid, model.predict(pool_valid))

    return quality