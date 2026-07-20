import optuna

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import mean_average_precision_at_k

def tuning_objective_als(trial, train_interaction_matrix, test_interaction_matrix, K):

    # Возможные значения гиперпараметров
    factors = trial.suggest_int('factors', 32, 256, step=32)
    regularization = trial.suggest_float('regularization', 1e-4, 0.1, log=True)
    iterations = trial.suggest_int('iterations', 10, 30, step=5)
    alpha = trial.suggest_float('alpha', 1, 100, log=True)

    # Обучение модели ALS
    als_model = AlternatingLeastSquares(
        factors=factors, # Размеры эмбеддингов
        iterations=iterations, # Кол-во итераций при обучении
        regularization=regularization, # Параметр регуляризации
        alpha=alpha, # Вес сигнала
        random_state=42
    )

    als_model.fit(train_interaction_matrix)

    # Оценка качества
    quality = mean_average_precision_at_k(
        als_model,
        train_interaction_matrix,
        test_interaction_matrix,
        K=K,
        show_progress=False
    )

    return quality

