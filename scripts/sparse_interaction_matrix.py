import numpy as np
from scipy.sparse import csr_matrix

def sparse_interaction_matrix(df):

    # Все уникальные покупатели и товары
    all_customers = df.customer_id.unique()
    all_articles = df.article_id.unique()

    # Словари для отображения индетификаторов в индексы
    customer_id_map = {
        customer_id: i 
        for i, customer_id in enumerate(all_customers)
    }
    article_id_map = {
        article_id: i 
        for i, article_id in enumerate(all_articles)
    }

    # Словари для отображения индексов в идентификаторы (обратные словари)
    customer_index_map = {
        i: customer_id
        for customer_id, i in customer_id_map.items()
    }
    article_index_map = {
        i: article_id
        for article_id, i in article_id_map.items()
    }

    # Индексы строк и столбцов для разреженной матрицы
    rows = df.customer_id.map(customer_id_map)
    cols = df.article_id.map(article_id_map)

    # Параметры матрицы
    interactions = np.ones(len(df)) # при наличии покупки = 1
    matrix_shape = ( # размер матрицы
        len(customer_id_map), # кол-во уникальных покупателей
        len(article_id_map) # колв-во уникальных товаров
    ) 

    # Создание матрицы взаимодействия в разреженном формате
    interaction_matrix = csr_matrix(
        (interactions, (rows, cols)), # данные и индкусы
        shape=matrix_shape # размер матрицы
    )

    return interaction_matrix, customer_id_map, article_id_map, customer_index_map, article_index_map
