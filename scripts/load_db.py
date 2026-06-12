import duckdb

def load_db(db_path='data/hnm_recommendations.db', data_path='data/raw'):

    # Предопределяем БД
    con = duckdb.connect(database=db_path)

    # Словарь с названиями таблиц и путями к данным
    datasets = {
        'transactions': f'{data_path}/transactions_train.parquet',
        'articles': f'{data_path}/articles.parquet',
        'customers': f'{data_path}/customers.parquet'
    }

    # Загружаем данные в БД
    for name, path in datasets.items():
        con.execute(f"""
           CREATE OR REPLACE TABLE {name} AS
           SELECT *
           FROM read_parquet('{path}') 
        """)

    return con