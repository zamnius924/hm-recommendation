import optuna

from catboost import CatBoostRanker
from scripts.map_at_k import map_at_k

def tuning_objective_ltr(trial, pool_train, pool_valid, df_valid):

    # Возможные значения гиперпараметров
    depth = trial.suggest_int('depth', 4, 12, step=1) # глубина деревьев
    #iterations = trial.suggest_int( # кол-во деревьев
    #    'iterations', 300, 1500, step=100
    #) 
    learning_rate = trial.suggest_float( # шаг в градиентном бустинге
        'learning_rate', 0.01, 0.2, log=True
    ) 
    l2_leaf_reg = trial.suggest_float( # регуляризация на листьях
        'l2_leaf_reg', 1, 100, log=True
    ) 

    params = {
        # Фиксированные параметры
        'loss_function': 'YetiRank',
        'random_seed': 42,
        'verbose': False,
        'iterations': 5000,
        # Калибруемые параметры
        'depth': depth,
        'learning_rate': learning_rate,
        'l2_leaf_reg': l2_leaf_reg,
        #'iterations': iterations
    }

    # Обучение модели LTR
    model = CatBoostRanker(**params)

    model.fit(
        pool_train,
        eval_set=pool_valid,
        use_best_model=True,
        early_stopping_rounds=200,
        verbose=False
    )

    # Оценка качества
    quality = map_at_k(df_valid, model.predict(pool_valid))

    return quality