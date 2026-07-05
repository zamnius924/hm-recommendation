def generate_features_customer(
        con, 
        split_dates: dict
    ):

    # Статистики по покупателю
    con.execute(f"""
        CREATE OR REPLACE TABLE customer_features AS
                
        SELECT
            t.customer_id,
            COUNT(*) AS num_purchases,
            COUNT(DISTINCT t.article_id) AS num_unique_articles,
            COUNT(DISTINCT t.t_dat) AS num_purchase_days,
            '{split_dates['feature_window_end']}' - MAX(t.t_dat) AS days_since_last_purchase,
            AVG(t.price) AS avg_price,
            MEDIAN(t.price) AS median_price,
            COALESCE(STDDEV(t.price), 0) AS std_price,
            MIN(t.price) AS min_price,
            MAX(t.price) AS max_price
        FROM transactions t
        LEFT JOIN customers c ON c.customer_id = t.customer_id
        WHERE t.t_dat BETWEEN 
            '{split_dates['feature_window_start']}' AND
            '{split_dates['feature_window_end']}'
        GROUP BY t.customer_id
    """)

    # Дополнительные характеристики покупателя
    con.execute(f"""
        CREATE OR REPLACE TABLE customer_features AS

        -- Вспомогательные статистики для заполнения пропусков  
        WITH stats AS (
            SELECT
                AVG(age) AS avg_age,
                MODE(fashion_news_frequency) AS mode_fashion_news_frequency,
                MODE(club_member_status) AS mode_club_member_status
            FROM customers
        )
        
        -- Добавление покупательских характеристик
        SELECT
            cf.*,
            COALESCE(c.age, (SELECT avg_age FROM stats)) AS customer_age,
            COALESCE(c.club_member_status, (SELECT mode_club_member_status FROM stats)) AS club_member_status,
            COALESCE(c.fashion_news_frequency, (SELECT mode_fashion_news_frequency FROM stats)) AS fashion_news_frequency
        FROM customer_features cf
        LEFT JOIN customers c ON cf.customer_id = c.customer_id
    """)
    