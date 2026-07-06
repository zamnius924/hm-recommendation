import logging
import pandas as pd

from scripts.build_logger import build_logger
from scripts.generate_features_article import generate_features_article
from scripts.generate_features_customer import generate_features_customer
from scripts.generate_pairs import generate_pairs
from scripts.row_counts import row_counts

def generate_tt_data(
        con, 
        split_dates: dict,
        return_df: bool = False
    ):
    
    # Инициализация логгера
    logger = build_logger()

    # Фичи: покупатели
    logger.info('[1/3] Create customer features')
    generate_features_customer(con, split_dates)
    row_counts(con, logger, 'customer_features')

    # Фичи: товары
    logger.info('[2/3] Create article features')
    generate_features_article(con, split_dates)
    row_counts(con, logger, 'article_features')

    # Таргет: positive samples
    logger.info('[3/3] Create target')
    generate_pairs(con, split_dates)
    row_counts(con, logger, 'customer_features')
    row_counts(con, logger, 'article_features')
    row_counts(con, logger, 'pairs')

    if return_df:
        # Создание датафреймов pandas
        df_customer = con.execute("""
            SELECT * FROM customer_features ORDER BY customer_id
        """).df()
        df_article = con.execute("""
            SELECT * FROM article_features ORDER BY article_id
        """).df()
        df_pairs = con.execute("""
            SELECT * FROM pairs ORDER BY customer_id, article_id
        """).df()

        # Тест 1: в df_pairs и df_customer одинаковые покупатели
        test_object_match(
            logger, 
            df_customer['customer_id'], 
            df_pairs['customer_id'], 
            type='customer', 
            num=1)

        # Тест 2: в df_pairs и df_article одинаковые товары
        test_object_match(
            logger, 
            df_article['article_id'], 
            df_pairs['article_id'], 
            type='article', 
            num=2)

        # Тест 3: в парах нет дубликатов
        test_duplicates(logger, df_pairs, 3)

        return df_customer, df_article, df_pairs


# ---------------------------------------------------------------------------- #
#                                     Тесты                                    #
# ---------------------------------------------------------------------------- #
def test_object_match(
        logger: logging.Logger, 
        object_id_features: pd.Series, 
        object_id_pairs: pd.DataFrame,
        type: str,
        num: int,
        total: int = 3
    ):
    """
    Тест: одинаковые объекты в df_* и df_pairs
    """
    
    # Уникальные объекты
    unique_el_features = set(object_id_features)
    unique_el_pairs = set(object_id_pairs)
    
    if unique_el_features == unique_el_pairs:
        logger.info(
            f'Test {num}/{total} passed: ID in pairs and {type}_features match '
            f'({len(unique_el_features)})'
        )
    else:
        logger.warning(
            f'Test {num}/{total} failed: ID in pairs and {type}_features differ '
            f'({len(unique_el_pairs)} vs {len(unique_el_features)})'
        )


def test_duplicates(
        logger: logging.Logger, 
        df_pairs: pd.DataFrame,
        num: int,
        total: int = 3
    ):
    """
    Тест: только уникальные пары
    """

    # Поиск дубликатов пар
    duplicates = df_pairs.duplicated().sum()

    if duplicates == 0:
        logger.info(f'Test {num}/{total} passed: pairs contain no duplicates')
    else:
        logger.warning(f'Test {num}/{total} failed: {duplicates} duplicate pairs found')