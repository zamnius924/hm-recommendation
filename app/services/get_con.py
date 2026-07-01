import duckdb

from scripts.paths import DATA_RAW_DIR, DATA_RECOMMENDATIONS_DIR

def get_con():

    # Предопределяем БД
    con = duckdb.connect(database=':memory:') # database=db_path

    # Словарь с названиями таблиц и путями к данным
    datasets = {
        'recommendations': DATA_RECOMMENDATIONS_DIR / 'df_rec.parquet',
        'articles': DATA_RAW_DIR / 'articles.parquet',
        'customers': DATA_RAW_DIR / 'customers.parquet'
    }
    
    # Загружаем данные в БД
    for name, path in datasets.items():
        
        con.execute(f"""
            CREATE OR REPLACE TABLE {name} AS
            SELECT *
            FROM read_parquet('{path.as_posix()}')
        """)

    return con