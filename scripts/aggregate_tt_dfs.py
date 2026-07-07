import logging
import numpy as np
import pandas as pd

from scripts.build_logger import build_logger
from scripts.class_tt_data import AggregateData, PairData, \
    FeatureData, Mapping
from sklearn.preprocessing import LabelEncoder

def aggregate_tt_dfs(
        df_customer: pd.DataFrame,
        df_article: pd.DataFrame,
        df_pairs: pd.DataFrame,
        mapping: dict
    ):

    # Инициализация логгера
    logger = build_logger()

    # ----------------------------- Смена индексации ----------------------------- #
    df_customer['customer_index'] = (df_customer['customer_id']
                                  .map(mapping['customer_id2index']))
    
    df_article['article_index'] = (df_article['article_id']
                                .map(mapping['article_id2index']))
    
    df_pairs['customer_index'] = (df_pairs['customer_id']
                               .map(mapping['customer_id2index']))
    
    df_pairs['article_index'] = (df_pairs['article_id']
                              .map(mapping['article_id2index']))
    
    # Тест 1: соблюдена сортировка по customer_index
    test(logger, df_customer, 'customer_index')

    # Тест 2: соблюдена сортировка по article_index
    test(logger, df_article, 'article_index')
    
    
    # ------------------------------ Правки к данным ----------------------------- #
    # Удаление индексов и идентификаторов
    df_customer = df_customer.drop(
        ['customer_id', 'customer_index'],
        axis=1
    )
    df_article = df_article.drop(
        ['article_id', 'article_index'],
        axis=1
    )
    df_pairs = df_pairs.drop(
        ['customer_id', 'article_id'],
        axis=1
    )
    
    # Удаление дат
    df_article = df_article.drop(
        df_article.select_dtypes(include='datetime').columns,
        axis=1
    )


    # --------------------------- Форматирование данных -------------------------- #
    # Формиатирование данных по покупателям и товарам
    customers = FeatureData(**df_to_data(df_customer))
    articles = FeatureData(**df_to_data(df_article))

    # Форматирование данных по парам
    pairs = PairData(
        customer_index=df_pairs['customer_index'].to_numpy(dtype=np.int64),
        article_index=df_pairs['article_index'].to_numpy(dtype=np.int64)
    )

    # Результирующие данные
    aggr_data = AggregateData(
        customers=customers,
        articles=articles,
        pairs=pairs,
        mapping=Mapping(**mapping)
    )

    return aggr_data
    

# ---------------------------------------------------------------------------- #
#                            Вспомогательные функции                           #
# ---------------------------------------------------------------------------- #
def df_to_data(df: pd.DataFrame):
    """
    Форматирование данных:
        - разделение признаков на числовые и категориальные
        - label-encoding для категориальных признаков
    """

    # Данные
    data_numeric = df.select_dtypes(include=['int', 'float'])
    data_categorical = df.select_dtypes(exclude=['int', 'float'])
    
    # Названия столбцов
    numeric_columns = data_numeric.columns

    categorical_columns = {}
    for col in data_categorical:
        
        # Инициализация энкодера
        le = LabelEncoder()
        
        # Применение энкодера
        data_categorical[col] = le.fit_transform(data_categorical[col])

        # Сохранение энкодера и кол-во классов
        categorical_columns[col] = {
            'encoder': le,
            'n_classes': len(le.classes_)
        }

    # Конвертация данных в массивы
    numeric = data_numeric.to_numpy(dtype=np.float32)
    categorical = data_categorical.to_numpy(dtype=np.int64)

    return {
        'numeric': numeric,
        'categorical': categorical,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
    }


# ---------------------------------------------------------------------------- #
#                                     Тесты                                    #
# ---------------------------------------------------------------------------- #
def test(logger: logging.Logger, df: list, col: str):
    """
    Тест: сортировка индексов в df_* корректная
    """
    
    ok = df[col].is_monotonic_increasing
    
    if ok:
        logger.info(f'Test passed: {col} are sorted correctly')
    else:
        logger.info(f'Test failed: {col} are sorted incorrectly')
