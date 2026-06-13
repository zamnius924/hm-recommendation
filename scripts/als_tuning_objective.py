import optuna

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k

def als_tuning_objective(trial, train_interaction_matrix, test_interaction_matrix):

    # Возможные значения гиперпараметров
    factors = trial.suggest_int('factors', 32, 256, step=32)
    regularization = trial.suggest_float('regularization', 1e-4, 0.1, log=True)
    iterations = trial.suggest_int('iterations', 10, 30, step=5)
    alpha = trial.suggest_float('alpha', 1, 100, log=True)

    # Обучение модели ALS
    als_model = AlternatingLeastSquares(
        factors=factors,  # Number of latent factors
        iterations=iterations,  # Number of iterations to train
        regularization=regularization, # Strength of regularisation parameter
        alpha=alpha,  # Confidence weighting factor
        random_state=42 # For reproducibility
    )

    als_model.fit(train_interaction_matrix)

    # Предсказание
    prec_at_10 = precision_at_k(
        als_model,
        train_interaction_matrix,
        test_interaction_matrix,
        K=10,
        show_progress=False
    )

    return prec_at_10

