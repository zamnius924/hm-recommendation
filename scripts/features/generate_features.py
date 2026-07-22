import _duckdb
import pandas as pd

from scripts.utils.build_logger import build_logger
from scripts.features.generate_features_article import generate_features_article
from scripts.features.generate_features_customer import generate_features_customer
from scripts.utils.row_counts import row_counts

def generate_features(
        con: _duckdb.DuckDBPyConnection, 
        split_dates: dict, 
        candidates: pd.DataFrame, 
        target: bool = True,
        return_con: bool = False
    ):

    # Инициализация логгера
    logger = build_logger()

    logger.info("=" * 80)
    logger.info("Starting feature generation")
    logger.info("=" * 80)

    # ------------------------ Добавление кандидатов в con ----------------------- #
    logger.info('[1/7] Register candidates')
    
    register_candidates(con, candidates)

    # Размер таблицы
    row_counts(con, logger, 'candidates')


    # ----------------------------- Фичи: покупатель ----------------------------- #
    logger.info('[2/7] Build customer features')
    
    generate_features_customer(con, split_dates)

    # Размер таблицы
    row_counts(con, logger, 'customer_features')


    # -------------------------------- Фичи: товар ------------------------------- #
    logger.info('[3/7] Build article features')
    
    generate_features_article(con, split_dates)

    # Размер таблицы
    row_counts(con, logger, 'article_features')


    # ------------------------- Фичи: покупатель-продукт ------------------------- #
    logger.info('[4/7] Build customer-article features')
    
    generate_features_customer_article(con, split_dates)

    # Размер таблицы
    row_counts(con, logger, 'candidates')


    # ------------- Объединение всех фичей + создание дополнительных ------------- #
    logger.info('[5/7] Merge features')
    
    # Объединение
    merge_features(con)

    # Дополнительные фичи
    generate_add_features(con)

    # Размер таблицы
    row_counts(con, logger, 'candidates')


    # ---------------------------------- Таргет ---------------------------------- #
    if target:
        logger.info('[6/7] Build target')

        generate_target(con, split_dates)
        
    else:
        logger.info('[6/7] Skip target')


    # ----------------------- Экспорт итогового датафрейма ----------------------- #
    if not return_con:

        logger.info('[7/7] Export dataframe')

        # Сортировка датафрейма и экспорт в pandas
        df = con.execute("""
            SELECT * 
            FROM candidates
            ORDER BY customer_id, article_id
        """).df()

        logger.info(f'candidates: %s rows', f'{df.shape[0]:,}')

        return df
    
    else:
        
        logger.info('[7/7] Export connector')

        # Сортировка датафрейма
        con.execute("""
            CREATE OR REPLACE TABLE candidates AS
            SELECT * 
            FROM candidates
            ORDER BY customer_id, article_id
        """)

        # Удаление других таблиц
        tables = con.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
        """).fetchall()

        for (table,) in tables:
            if table != 'candidates':
                con.execute(f"DROP TABLE IF EXISTS {table}")

        return con
    



# ---------------------------------------------------------------------------- #
#                            Вспомогательные функции                           #
# ---------------------------------------------------------------------------- #
def register_candidates(
        con: _duckdb.DuckDBPyConnection,
        candidates: pd.DataFrame
    ):
    """
    Добавление кандидатов в БД
    """
    
    # Инициализация временной таблицы
    con.register("candidates_df", candidates)

    # Запись таблицы в БД
    con.execute("""
        CREATE OR REPLACE TABLE candidates AS
        SELECT *
        FROM candidates_df
    """)

    # Удаление вспомогательной таблицы candidates_df
    con.unregister("candidates_df")


def generate_features_customer_article(
        con: _duckdb.DuckDBPyConnection,
        split_dates: dict,
        max_days: int = 9999 # кол-во дней с покупки товара, который не был куплен
    ):
    """
    Генерация фичей покупатель-продукт
    """

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE history_base AS
        SELECT 
            t.customer_id,
            t.article_id,
            t.t_dat,
            a.product_type_name,
            a.department_name,
            a.product_group_name,
            a.colour_group_name,
            a.garment_group_name
        FROM transactions t
        LEFT JOIN articles a ON t.article_id = a.article_id
        WHERE t.t_dat <= '{split_dates['feature_window_end']}'
          AND t.customer_id IN (SELECT DISTINCT customer_id FROM candidates)
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE candidates AS        

        -- Article-level
        WITH customer_article_history AS (
            SELECT 
                customer_id,
                article_id,
                (DATE('{split_dates['feature_window_end']}') - MAX(t_dat)) AS days_since_last_article_purchase,
                COUNT(*) AS count_same_article_purchases
            FROM history_base
            WHERE t_dat <= '{split_dates['feature_window_end']}'
            GROUP BY customer_id, article_id
        ), 
        
        -- Type-level
        customer_type_history AS (
            SELECT
                customer_id,
                product_type_name,
                (DATE('{split_dates['feature_window_end']}') - MAX(t_dat)) AS days_since_last_type_purchase,
                COUNT(*) AS count_same_type_purchases
            FROM history_base
            WHERE t_dat <= '{split_dates['feature_window_end']}'
            GROUP BY customer_id, product_type_name
        ), 
        
        -- Department-level
        customer_department_history AS (
            SELECT
                customer_id,
                department_name,
                (DATE('{split_dates['feature_window_end']}') - MAX(t_dat)) AS days_since_last_department_purchase,
                COUNT(*) AS count_same_department_purchases
            FROM history_base
            WHERE t_dat <= '{split_dates['feature_window_end']}'
            GROUP BY customer_id, department_name
        ), 
        
        -- Group-level
        customer_group_history AS (
            SELECT
                customer_id,
                product_group_name,
                (DATE('{split_dates['feature_window_end']}') - MAX(t_dat)) AS days_since_last_group_purchase,
                COUNT(*) AS count_same_group_purchases
            FROM history_base
            WHERE t_dat <= '{split_dates['feature_window_end']}'
            GROUP BY customer_id, product_group_name
        ), 
        
        -- Colour-level
        customer_colour_history AS (
            SELECT
                customer_id,
                colour_group_name,
                (DATE('{split_dates['feature_window_end']}') - MAX(t_dat)) AS days_since_last_colour_purchase,
                COUNT(*) AS count_same_colour_purchases
            FROM history_base
            WHERE t_dat <= '{split_dates['feature_window_end']}'
            GROUP BY customer_id, colour_group_name
        ), 
        
        -- Garment-level
        customer_garment_history AS (
            SELECT
                customer_id,
                garment_group_name,
                (DATE('{split_dates['feature_window_end']}') - MAX(t_dat)) AS days_since_last_garment_purchase,
                COUNT(*) AS count_same_garment_purchases
            FROM history_base
            WHERE t_dat <= '{split_dates['feature_window_end']}'
            GROUP BY customer_id, garment_group_name
        )

        SELECT
            can.*,

            -- Индикаторы покупок
            CASE WHEN h_a.customer_id IS NOT NULL THEN 1 ELSE 0 END AS customer_bought_article_before,
            CASE WHEN h_t.customer_id IS NOT NULL THEN 1 ELSE 0 END AS customer_bought_type_before,
            CASE WHEN h_d.customer_id IS NOT NULL THEN 1 ELSE 0 END AS customer_bought_department_before,
            CASE WHEN h_gr.customer_id IS NOT NULL THEN 1 ELSE 0 END AS customer_bought_group_before,
            CASE WHEN h_c.customer_id IS NOT NULL THEN 1 ELSE 0 END AS customer_bought_colour_before,
            CASE WHEN h_ga.customer_id IS NOT NULL THEN 1 ELSE 0 END AS customer_bought_garment_before,
            
            -- Количество дней с поледней покупки
            COALESCE(h_a.days_since_last_article_purchase, {max_days}) AS days_since_last_article_purchase,
            COALESCE(h_t.days_since_last_type_purchase, {max_days}) AS days_since_last_type_purchase,
            COALESCE(h_d.days_since_last_department_purchase, {max_days}) AS days_since_last_department_purchase,
            COALESCE(h_gr.days_since_last_group_purchase, {max_days}) AS days_since_last_group_purchase,
            COALESCE(h_c.days_since_last_colour_purchase, {max_days}) AS days_since_last_colour_purchase,
            COALESCE(h_ga.days_since_last_garment_purchase, {max_days}) AS days_since_last_garment_purchase,

            -- Количество покупок
            COALESCE(h_a.count_same_article_purchases, 0) AS count_same_article_purchases,
            COALESCE(h_t.count_same_type_purchases, 0) AS count_same_type_purchases,
            COALESCE(h_d.count_same_department_purchases, 0) AS count_same_department_purchases,
            COALESCE(h_gr.count_same_group_purchases, 0) AS count_same_group_purchases,
            COALESCE(h_c.count_same_colour_purchases, 0) AS count_same_colour_purchases,
            COALESCE(h_ga.count_same_garment_purchases, 0) AS count_same_garment_purchases

        FROM candidates can
        LEFT JOIN articles a 
            ON can.article_id = a.article_id
        LEFT JOIN customer_article_history h_a
            ON can.customer_id = h_a.customer_id
            AND a.article_id = h_a.article_id
        LEFT JOIN customer_type_history h_t
            ON can.customer_id = h_t.customer_id
            AND a.product_type_name = h_t.product_type_name
        LEFT JOIN customer_department_history h_d
            ON can.customer_id = h_d.customer_id
            AND a.department_name = h_d.department_name
        LEFT JOIN customer_group_history h_gr
            ON can.customer_id = h_gr.customer_id
            AND a.product_group_name = h_gr.product_group_name
        LEFT JOIN customer_colour_history h_c
            ON can.customer_id = h_c.customer_id
            AND a.colour_group_name = h_c.colour_group_name
        LEFT JOIN customer_garment_history h_ga
            ON can.customer_id = h_ga.customer_id
            AND a.garment_group_name = h_ga.garment_group_name
    """)


def merge_features(
        con: _duckdb.DuckDBPyConnection
    ):
    """
    Сводная таблица (объединение таблиц: 
    1) фичи покупателей, 2) товаров, 3) покупателей-товаров)
    """

    con.execute(f"""
        CREATE OR REPLACE TABLE candidates AS 
        SELECT
            can.*,
            af.* EXCLUDE (article_id),
            cf.* EXCLUDE (customer_id)
        FROM candidates can
        LEFT JOIN article_features af ON can.article_id = af.article_id
        LEFT JOIN customer_features cf ON can.customer_id = cf.customer_id
    """)


def generate_add_features(
        con: _duckdb.DuckDBPyConnection
    ):
    """
    Генерация дополнительных фичей на базе сводной таблицы
    """

    con.execute(f"""
        CREATE OR REPLACE TABLE candidates AS 
        SELECT
            *,
            COALESCE((article_avg_price - avg_price) / NULLIF(std_price, 0), 0) AS price_zscore,
            article_avg_price / NULLIF(median_price, 0) AS price_ratio_median,
            ABS(article_avg_price - median_price) AS price_distance_median,
            customer_age - article_avg_customer_age AS dev_customer_age
        FROM candidates can
    """)


def generate_target(
        con: _duckdb.DuckDBPyConnection,
        split_dates: dict
    ):
    """
    Создание таргета
    """

    con.execute(f"""
        CREATE OR REPLACE TABLE candidates AS         

        WITH target AS (
            SELECT DISTINCT customer_id, article_id
            FROM transactions
            WHERE t_dat BETWEEN 
                '{split_dates['target_window_start']}' AND
                '{split_dates['target_window_end']}'
        )

        SELECT 
            can.*,
            CASE
                WHEN t.customer_id IS NOT NULL THEN 1
                ELSE 0
            END AS target
        FROM candidates can
        LEFT JOIN target t 
            ON can.customer_id = t.customer_id
            AND can.article_id = t.article_id     
    """)

