import duckdb

def load_db(data_path='data/raw'): # db_path='data/hnm_recommendations.db'

    # Предопределяем БД
    con = duckdb.connect(database=':memory:') # database=db_path

    # Словарь с названиями таблиц и путями к данным
    datasets = {
        'transactions': f'{data_path}/transactions_train.parquet',
        'articles': f'{data_path}/articles.parquet',
        'customers': f'{data_path}/customers.parquet'
    }

    # Загружаем данные в БД
    for name, path in datasets.items():
        query_start = f"CREATE OR REPLACE TABLE {name} AS "
        query_mid = "SELECT * "
        query_end = f"FROM read_parquet('{path}') "

        if name == 'transactions':
            # В таблице transactions меняем тип t_dat на DATE
            query_mid = "SELECT CAST(t_dat AS DATE) AS t_dat, * EXCLUDE (t_dat)"
    
        con.execute(f"""
            {query_start}{query_mid}
            {query_end}
        """)

    return con