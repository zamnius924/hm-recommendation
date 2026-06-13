import numpy as np
from scipy.sparse import csr_matrix

def sparse_interaction_matrix(df, customer_id2index, article_id2index):

    # Индексы строк и столбцов для разреженной матрицы
    rows = df.customer_id.map(customer_id2index)
    cols = df.article_id.map(article_id2index)

    # Параметры матрицы
    interactions = np.ones(len(df)) # при наличии покупки = 1
    matrix_shape = ( # размер матрицы
        len(customer_id2index), # кол-во уникальных покупателей
        len(article_id2index) # колв-во уникальных товаров
    ) 

    # Создание матрицы взаимодействия в разреженном формате
    interaction_matrix = csr_matrix(
        (interactions, (rows, cols)), # данные и индкусы
        shape=matrix_shape # размер матрицы
    )

    return interaction_matrix
