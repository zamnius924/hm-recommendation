def generate_features_article(con, split_dates: dict):

    # Сатистики по товару
    con.execute(f"""
        CREATE OR REPLACE TABLE article_features AS
        SELECT 
            article_id,
            COUNT(*) AS article_purchase_count,
            COUNT(DISTINCT customer_id) AS article_unique_customers,
            COUNT(*) FILTER (
                WHERE t_dat > (DATE('{split_dates['feature_window_end']}') - INTERVAL '7 days')
            ) AS article_purchase_count_7d,
            COUNT(*) FILTER (
                WHERE t_dat > (DATE('{split_dates['feature_window_end']}') - INTERVAL '14 days')
            ) AS article_purchase_count_14d,
            COUNT(*) FILTER (
                WHERE t_dat > (DATE('{split_dates['feature_window_end']}') - INTERVAL '30 days')
            ) AS article_purchase_count_30d,
            AVG(price) AS article_avg_price,
            AVG(ABS(sales_channel_id - 2)) AS online_purchased_rate
        FROM transactions
        WHERE t_dat BETWEEN 
            '{split_dates['feature_window_start']}' AND
            '{split_dates['feature_window_end']}'
        GROUP BY article_id
    """)

    # Возрастные фичи
    con.execute(f"""
        CREATE OR REPLACE TABLE article_features AS

        -- Возраст товара
        WITH article_age AS (
            SELECT
                t.article_id,
                MIN(t.t_dat) AS article_first_purchase_date, -- Дата первой продажи товара
                MAX(t.t_dat) AS article_last_purchase_date, -- Дата последней продажи товара
                (DATE('{split_dates['feature_window_end']}') - MIN(t.t_dat)) AS article_days_since_first_purchase, -- Кол-во дней с первой покупки
                (DATE('{split_dates['feature_window_end']}') - MAX(t.t_dat)) AS article_days_since_last_purchase -- Кол-во дней с последней покупки
            FROM transactions t
            WHERE t.t_dat <= '{split_dates['feature_window_end']}'
            GROUP BY t.article_id
        ),

        -- Возраст покупателей
        customer_age AS (
            SELECT
                t.article_id,
                AVG(c.age) AS article_avg_customer_age, -- Средний возраст покупателей товара
                COUNT(*) FILTER (WHERE c.age > 18 AND c.age <= 24) AS article_purchase_count_18_24,
                COUNT(*) FILTER (WHERE c.age > 24 AND c.age <= 34) AS article_purchase_count_24_34,
                COUNT(*) FILTER (WHERE c.age > 34 AND c.age <= 44) AS article_purchase_count_34_44,
                COUNT(*) FILTER (WHERE c.age > 44 AND c.age <= 54) AS article_purchase_count_44_54,
                COUNT(*) FILTER (WHERE c.age > 54 AND c.age <= 64) AS article_purchase_count_54_64,
                COUNT(*) FILTER (WHERE c.age > 64) AS article_purchase_count_64_99
            FROM transactions t
            LEFT JOIN customers c ON c.customer_id = t.customer_id
            WHERE t_dat BETWEEN 
                '{split_dates['feature_window_start']}' AND
                '{split_dates['feature_window_end']}'
            GROUP BY article_id 
        )

        SELECT 
            f.*,
            a.* EXCLUDE (a.article_id),
            c.article_avg_customer_age,
            c.article_purchase_count_18_24 / NULLIF(f.article_purchase_count, 0) AS article_purchase_share_18_24,
            c.article_purchase_count_24_34 / NULLIF(f.article_purchase_count, 0) AS article_purchase_share_24_34,
            c.article_purchase_count_34_44 / NULLIF(f.article_purchase_count, 0) AS article_purchase_share_34_44,
            c.article_purchase_count_44_54 / NULLIF(f.article_purchase_count, 0) AS article_purchase_share_44_54,
            c.article_purchase_count_54_64 / NULLIF(f.article_purchase_count, 0) AS article_purchase_share_54_64,
            c.article_purchase_count_64_99 / NULLIF(f.article_purchase_count, 0) AS article_purchase_share_64_99
        FROM article_features f
        LEFT JOIN article_age a ON f.article_id = a.article_id
        LEFT JOIN customer_age c ON f.article_id = c.article_id
    """)
    