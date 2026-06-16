# %%  Импорт библиотек
import datetime as dt
import duckdb
import json
import pandas as pd

from scripts.load_db import load_db

# %% Импорт
# Подключение к БД и загрузка данных
con = load_db()

# Добавление датасета с ALS-рекомендациями в con
con.execute("""
    CREATE OR REPLACE TABLE als_candidates AS
    SELECT *
    FROM read_parquet('data/processed/als_candidates.parquet')
""")

# Временное разделение на train- и test-выборки
with open(file='data/processed/split_dates.json', mode='r') as file:
    dates = json.load(file)

# Проверка таблиц
con.execute("SHOW TABLES").df()

# %% Customer features
con.execute(f"""
    CREATE OR REPLACE TABLE customer_features AS
    SELECT
        t.customer_id,
        COUNT(*) AS num_purchases,
        COUNT(DISTINCT t.article_id) AS num_unique_articles,
        COUNT(DISTINCT t.t_dat) AS num_purchase_days,
        '{dates['train_end']}' - MAX(t.t_dat) AS days_since_last_purchase,
        AVG(t.price) AS avg_price,
        MEDIAN(t.price) AS median_price,
        COALESCE(STDDEV(t.price), 0) AS std_price,
        MIN(t.price) AS min_price,
        MAX(t.price) AS max_price
    FROM transactions t
    LEFT JOIN customers c ON c.customer_id = t.customer_id
    WHERE t.t_dat BETWEEN '{dates['train_start']}' AND '{dates['train_end']}'
    GROUP BY t.customer_id
""")

# %% Article features
con.execute(f"""
    CREATE OR REPLACE TABLE article_features AS
    SELECT 
        article_id,
        COUNT(*) AS article_purchase_count,
        COUNT(DISTINCT customer_id) AS article_unique_customers,
        COUNT(*) FILTER (
            WHERE t_dat > (DATE('{dates['train_end']}') - INTERVAL '7 days')
        ) AS article_purchase_count_7d,
        COUNT(*) FILTER (
            WHERE t_dat > (DATE('{dates['train_end']}') - INTERVAL '14 days')
        ) AS article_purchase_count_14d,
        COUNT(*) FILTER (
            WHERE t_dat > (DATE('{dates['train_end']}') - INTERVAL '30 days')
        ) AS article_purchase_count_30d,
        AVG(price) AS article_avg_price
    FROM transactions
    WHERE t_dat BETWEEN '{dates['train_start']}' AND '{dates['train_end']}'
    GROUP BY article_id
""")

# %% Article age
con.execute(f"""
    CREATE OR REPLACE TABLE article_features AS
            
    WITH article_age AS (
        SELECT
            article_id,
            MIN(t_dat) AS article_first_purchase_date,
            (DATE('{dates['train_end']}') - MIN(t_dat)) AS article_age
        FROM transactions
        WHERE t_dat <= '{dates['train_end']}'
        GROUP BY article_id
    )

    SELECT f.*, a.* EXCLUDE (a.article_id)
    FROM article_features f
    LEFT JOIN article_age a ON f.article_id = a.article_id
""")

# %% Customer-article features
con.execute(f"""
    CREATE OR REPLACE TABLE als_candidates AS        
    
    WITH customer_article_history AS (
        SELECT 
            customer_id,
            article_id,
            (DATE('{dates['train_end']}') - MAX(t_dat)) AS days_since_last_article_purchase,
            COUNT(*) AS count_same_article_purchases
        FROM transactions
        WHERE t_dat <= '{dates['train_end']}'
        GROUP BY customer_id, article_id
    ), customer_type_history AS (
        SELECT
            t.customer_id,
            a.product_type_name,
            (DATE('{dates['train_end']}') - MAX(t.t_dat)) AS days_since_last_type_purchase,
            COUNT(*) AS count_same_type_purchases
        FROM transactions t
        LEFT JOIN articles a ON t.article_id = a.article_id
        WHERE t_dat <= '{dates['train_end']}'
        GROUP BY t.customer_id, a.product_type_name
    ), customer_department_history AS (
        SELECT
            t.customer_id,
            a.department_name,
            (DATE('{dates['train_end']}') - MAX(t.t_dat)) AS days_since_last_department_purchase,
            COUNT(*) AS count_same_department_purchases
        FROM transactions t
        LEFT JOIN articles a ON t.article_id = a.article_id
        WHERE t_dat <= '{dates['train_end']}'
        GROUP BY t.customer_id, a.department_name
    ), customer_group_history AS (
        SELECT
            t.customer_id,
            a.product_group_name,
            (DATE('{dates['train_end']}') - MAX(t.t_dat)) AS days_since_last_group_purchase,
            COUNT(*) AS count_same_group_purchases
        FROM transactions t
        LEFT JOIN articles a ON t.article_id = a.article_id
        WHERE t_dat <= '{dates['train_end']}'
        GROUP BY t.customer_id, a.product_group_name
    ), customer_colour_history AS (
        SELECT
            t.customer_id,
            a.colour_group_name,
            (DATE('{dates['train_end']}') - MAX(t.t_dat)) AS days_since_last_colour_purchase,
            COUNT(*) AS count_same_colour_purchases
        FROM transactions t
        LEFT JOIN articles a ON t.article_id = a.article_id
        WHERE t_dat <= '{dates['train_end']}'
        GROUP BY t.customer_id, a.colour_group_name
    ), customer_garment_history AS (
        SELECT
            t.customer_id,
            a.garment_group_name,
            (DATE('{dates['train_end']}') - MAX(t.t_dat)) AS days_since_last_garment_purchase,
            COUNT(*) AS count_same_garment_purchases
        FROM transactions t
        LEFT JOIN articles a ON t.article_id = a.article_id
        WHERE t_dat <= '{dates['train_end']}'
        GROUP BY t.customer_id, a.garment_group_name
    )
    
    SELECT
        als.*,
        CASE
            WHEN h_a.customer_id IS NOT NULL THEN 1
            ELSE 0
        END AS customer_bought_article_before,
        CASE
            WHEN h_t.customer_id IS NOT NULL THEN 1
            ELSE 0
        END AS customer_bought_type_before,
        CASE
            WHEN h_d.customer_id IS NOT NULL THEN 1
            ELSE 0
        END AS customer_bought_department_before,
        CASE
            WHEN h_gr.customer_id IS NOT NULL THEN 1
            ELSE 0
        END AS customer_bought_group_before,
        CASE
            WHEN h_c.customer_id IS NOT NULL THEN 1
            ELSE 0
        END AS customer_bought_colour_before,
        CASE
            WHEN h_ga.customer_id IS NOT NULL THEN 1
            ELSE 0
        END AS customer_bought_garment_before,
        h_a.days_since_last_article_purchase,
        h_t.days_since_last_type_purchase,
        h_d.days_since_last_department_purchase,
        h_gr.days_since_last_group_purchase,
        h_c.days_since_last_colour_purchase,
        h_ga.days_since_last_garment_purchase,
        h_a.count_same_article_purchases,
        h_t.count_same_type_purchases,
        h_d.count_same_department_purchases,
        h_gr.count_same_group_purchases,
        h_c.count_same_colour_purchases,
        h_ga.count_same_garment_purchases
    FROM als_candidates als
    LEFT JOIN articles a 
        ON als.article_id = a.article_id
    LEFT JOIN customer_article_history h_a
        ON als.customer_id = h_a.customer_id
        AND a.article_id = h_a.article_id
    LEFT JOIN customer_type_history h_t
        ON als.customer_id = h_t.customer_id
        AND a.product_type_name = h_t.product_type_name
    LEFT JOIN customer_department_history h_d
        ON als.customer_id = h_d.customer_id
        AND a.department_name = h_d.department_name
    LEFT JOIN customer_group_history h_gr
        ON als.customer_id = h_gr.customer_id
        AND a.product_group_name = h_gr.product_group_name
    LEFT JOIN customer_colour_history h_c
        ON als.customer_id = h_c.customer_id
        AND a.colour_group_name = h_c.colour_group_name
    LEFT JOIN customer_garment_history h_ga
        ON als.customer_id = h_ga.customer_id
        AND a.garment_group_name = h_ga.garment_group_name
""")

# %% Финальный датафрейм с дополнительными customer-article features
con.execute(f"""
    CREATE OR REPLACE TABLE als_candidates AS 
    SELECT
        als.*,
        af.* EXCLUDE (article_id),
        cf.* EXCLUDE (customer_id),
        COALESCE(
            (af.article_avg_price - cf.avg_price) / NULLIF(cf.std_price, 0),
            0
        ) AS price_zscore,
        af.article_avg_price / NULLIF(cf.median_price, 0) AS price_ratio_median,
        ABS(af.article_avg_price - cf.median_price) AS price_distance_median
    FROM als_candidates als
    LEFT JOIN article_features af ON als.article_id = af.article_id
    LEFT JOIN customer_features cf ON als.customer_id = cf.customer_id
""")

# %% Создание таргета
df = con.execute(f"""
    WITH target AS (
        SELECT DISTINCT customer_id, article_id
        FROM transactions
        WHERE t_dat BETWEEN '{dates['test_start']}' AND '{dates['test_end']}'        
    )

    SELECT 
        als.*,
        CASE
            WHEN t.customer_id IS NOT NULL THEN 1
            ELSE 0
        END AS target
    FROM als_candidates als
    LEFT JOIN target t 
        ON als.customer_id = t.customer_id
        AND als.article_id = t.article_id     
""").df()

# %% Сохранение датасета
df.to_parquet('data/features/ranking_features.parquet', 
              engine='pyarrow', index=False)

con.close()
