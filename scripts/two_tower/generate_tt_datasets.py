import logging
import pandas as pd

from scripts.data.build_mapping import build_mapping
from scripts.data.load_db import load_db
from scripts.features.generate_features_article import generate_features_article
from scripts.features.generate_features_customer import generate_features_customer
from scripts.two_tower.class_tt_data import TwoTowerDataset, TowerInfo ,\
    CustomerDataset, ArticleDataset
from scripts.utils.aggregate_tt_dfs import aggregate_tt_dfs
from scripts.utils.build_logger import build_logger
from scripts.utils.row_counts import row_counts

def generate_tt_train(
        split_dates: dict
    ) -> tuple[TwoTowerDataset, TowerInfo]:

    # --------------- Создание датафреймов с фичами и индексами пар -------------- #
    # Подключение к БД
    con = load_db()

    # Датафреймы с фичами и индексами пар
    df_customer, df_article, df_pairs = generate_tt_dfs(con, 
                                                        split_dates, 
                                                        return_df=True,
                                                        pairs=True)

    # Отключение от БД
    con.close()


    # ----------------- Форматирование данных для Two-tower model ---------------- #
    # Создание мэппинга
    mapping = build_mapping(df_customer, df_article, df_pairs)

    # Вспомогательные данные
    aggr_data = aggregate_tt_dfs(df_customer, 
                                 df_article, 
                                 df_pairs, 
                                 mapping,
                                 pairs=True)

    # Данные для Two-tower model
    tt_info = TowerInfo(aggr_data)
    tt_dataset = TwoTowerDataset(aggr_data)

    return tt_dataset, tt_info


def generate_tt_inference(
        split_dates: dict
    ) -> tuple[CustomerDataset, ArticleDataset, dict]:

    # --------------- Создание датафреймов с фичами и индексами пар -------------- #
    # Подключение к БД
    con = load_db()

    # Датафреймы с фичами и индексами пар
    df_customer, df_article = generate_tt_dfs(con, 
                                              split_dates, 
                                              return_df=True,
                                              pairs=False)

    # Отключение от БД
    con.close()

    # ----------------- Форматирование данных для Two-tower model ---------------- #
    # Создание мэппинга
    mapping = build_mapping(df_customer, df_article)

    # Вспомогательные данные
    aggr_data = aggregate_tt_dfs(df_customer, 
                                 df_article, 
                                 None, 
                                 mapping,
                                 pairs=False)
    
    customer_dataset = CustomerDataset(aggr_data.customers)
    article_dataset = ArticleDataset(aggr_data.articles)

    return customer_dataset, article_dataset, mapping




# ---------------------------------------------------------------------------- #
#                            Вспомогательные функции                           #
# ---------------------------------------------------------------------------- #
def generate_tt_dfs(
        con, 
        split_dates: dict,
        return_df: bool = False,
        pairs: bool = True
    ):
    
    # Инициализация логгера
    logger = build_logger()

    # Фичи
    generate_tt_dfs_features(con, logger, split_dates)

    if pairs:
        # Таргет: positive samples
        generate_tt_pairs(con, logger, split_dates)

    # Проверки
    row_counts(con, logger, 'customer_features')
    row_counts(con, logger, 'article_features')
    
    if pairs:
        row_counts(con, logger, 'pairs')

    if return_df:
        # Создание датафреймов pandas
        df_customer, df_article = load_tt_dfs_features(con)
        
        if pairs:

            # Создание датафреймов pandas
            df_pairs = load_tt_dfs_pairs(con)
        
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

        else:

            return df_customer, df_article


def generate_tt_dfs_features(
        con, 
        logger, 
        split_dates: dict
    ):

    # Фичи: покупатели
    logger.info('1. Create customer features')
    generate_features_customer(con, split_dates)
    row_counts(con, logger, 'customer_features')

    # Фичи: товары
    logger.info('2. Create article features')
    generate_features_article(con, split_dates)
    row_counts(con, logger, 'article_features')


def generate_tt_pairs(
        con, 
        logger,
        split_dates: dict
    ):
    """
    Генерация пар для in-batch negative sampling:
        - pairs хранит в себе информацию только о тех покупателях/товарах,
          которые присутствуют и в target_window и в feature_window
        - customer_features и article_features отфильтрованы так,
          чтобы объекты присутствовали в target_window
    """

    logger.info('3. Create pairs')

    # Создание positive pairs
    con.execute(f"""
        CREATE OR REPLACE TABLE pairs AS 

        SELECT DISTINCT 
            t.customer_id,
            t.article_id
        FROM transactions t
        INNER JOIN customer_features c ON c.customer_id = t.customer_id
        INNER JOIN article_features a ON a.article_id = t.article_id
        WHERE t.t_dat BETWEEN 
            '{split_dates['target_window_start']}' AND
            '{split_dates['target_window_end']}'
        ORDER BY t.customer_id, t.article_id
    """)

    # Оставялем объекты, которые есть в pairs
    # Покупатели
    con.execute(f"""
        CREATE OR REPLACE TABLE customer_features AS
        
        SELECT *
        FROM customer_features
        WHERE customer_id IN (SELECT DISTINCT customer_id FROM pairs)
    """)

    # Товары
    con.execute(f"""
        CREATE OR REPLACE TABLE article_features AS
        
        SELECT *
        FROM article_features
        WHERE article_id IN (SELECT DISTINCT article_id FROM pairs)
    """)


def load_tt_dfs_features(con):

    # Создание датафреймов pandas
    df_customer = con.execute("""
        SELECT * FROM customer_features ORDER BY customer_id
    """).df()
    df_article = con.execute("""
        SELECT * FROM article_features ORDER BY article_id
    """).df()

    return df_customer, df_article


def load_tt_dfs_pairs(con):

    # Создание датафреймов pandas
    df_pairs = con.execute("""
        SELECT * FROM pairs ORDER BY customer_id, article_id
    """).df()

    return df_pairs




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