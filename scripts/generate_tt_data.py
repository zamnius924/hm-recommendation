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
        unique_customers_1 = set(df_customer.customer_id)
        unique_customers_2 = set(df_pairs.customer_id)

        if unique_customers_1 == unique_customers_2:
            logger.info(
                'Test 1/3 passed: customers in pairs and customer_features match '
                f'({len(unique_customers_1)})'
            )
        else:
            logger.warning(
                'Test 1/3 failed: customers in pairs and customer_features differ '
                f'({len(unique_customers_2)} vs {len(unique_customers_1)})'
            )

        # Тест 2: в df_pairs и df_article одинаковые товары
        unique_articles_1 = set(df_article.article_id)
        unique_articles_2 = set(df_pairs.article_id)

        if unique_articles_1 == unique_articles_2:
            logger.info(
                'Test 2/3 passed: articles in pairs and article_features match '
                f'({len(unique_articles_1)})'
            )
        else:
            logger.warning(
                'Test 2/3 failed: articles in pairs and article_features differ '
                f'({len(unique_articles_2)} vs {len(unique_articles_1)})'
            )

        # Тест 3: в парах нет дубликатов
        duplicates = df_pairs.duplicated().sum()

        if duplicates == 0:
            logger.info('Test 3/3 passed: pairs contain no duplicates')
        else:
            logger.warning(f'Test 3/3 failed: {duplicates} duplicate pairs found')

        return df_customer, df_article, df_pairs
