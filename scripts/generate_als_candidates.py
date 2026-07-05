import numpy as np
import pandas as pd

from implicit.als import AlternatingLeastSquares
from scripts.build_mapping import build_mapping
from scripts.sparse_interaction_matrix import sparse_interaction_matrix
from scripts.window_extraction import window_extraction

def generate_als_candidates(
        con,
        split_dates: dict,
        als_params: dict,
        target: bool = True
    ):

    # ------------------------- Предопределение объектов ------------------------- #
    # Выделение target- и feature-window
    df_feature, df_target = window_extraction(con, split_dates, target)

    # Создание мэппинга для feature-window
    mapping = build_mapping(df_feature)

    # Создание матрицы взаимодействия для feature-window
    feature_interaction_matrix = sparse_interaction_matrix(
        df_feature,
        mapping['customer_id2index'],
        mapping['article_id2index']
    )


    # ---------------------------- Обучение модели ALS --------------------------- #
    # Инициализация модели
    als_model = AlternatingLeastSquares(
        factors=als_params['factors'], # Размеры эмбеддингов
        iterations=als_params['iterations'], # Кол-во итераций при обучении
        regularization=als_params['regularization'], # Параметр регуляризации
        alpha=als_params['alpha'], # Вес сигнала
        random_state=42
    )

    # Обучение модели
    als_model.fit(feature_interaction_matrix)


    # ---------- Дополнительные ограничения на feature- и target-window ---------- #
    if target:
        # target-window: только те, кто был в feature-window
        known_customers = mapping['customer_id2index'].keys()
        known_articles = mapping['article_id2index'].keys()

        df_target = df_target[
            df_target['customer_id'].isin(known_customers)
            & df_target['article_id'].isin(known_articles)
        ]

        # feature-window: только те, кто совершили покупку в target-window
        active_customers = set(df_target.customer_id)
    else:
        # активными считаются все пользователи в feature-window
        active_customers = set(df_feature.customer_id)
    
    customer_index = [ # Индексы покупателей, для которых строим рекомендации
        mapping['customer_id2index'][customer]
        for customer in active_customers
    ]


    # ----------------- Построение рекомендаций на target-window ----------------- #
    # Индексы и оценки рекомендуемых товаров
    article_index, article_score = als_model.recommend(
        userid=customer_index,
        user_items=feature_interaction_matrix[customer_index,:],
        N=als_params['K']
    )

    # Таблица с индексами покупателей и товаров
    als_candidates = pd.DataFrame({
        'customer_id': np.repeat(customer_index, als_params['K']),
        'article_id': article_index.ravel(),
        'article_score': article_score.ravel()
    })

    # Перевод индексов покупателей и товаров в идентификаторы
    als_candidates.customer_id = als_candidates.customer_id.map(mapping['customer_index2id'])
    als_candidates.article_id = als_candidates.article_id.map(mapping['article_index2id'])

    return als_candidates