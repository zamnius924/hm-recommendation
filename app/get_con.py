import duckdb

def get_con():

    # Предопределяем БД
    con = duckdb.connect(database=':memory:') # database=db_path

    # Словарь с названиями таблиц и путями к данным
    datasets = {
        'recommendations': 'data/recommendations/df_rec.parquet',
        'articles': 'data/raw/articles.parquet',
        'customers': 'data/raw/customers.parquet'
    }
    
    # Загружаем данные в БД
    for name, path in datasets.items():
        
        con.execute(f"""
            CREATE OR REPLACE TABLE {name} AS
            SELECT *
            FROM read_parquet('{path}')
        """)

    return con