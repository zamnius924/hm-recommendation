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
    