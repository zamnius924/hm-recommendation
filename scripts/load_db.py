import duckdb

from pathlib import Path
from scripts.paths import DATA_RAW_DIR

def load_db(data_path: Path = DATA_RAW_DIR): # db_path='data/hnm_recommendations.db'

    # Предопределяем БД
    con = duckdb.connect(database=':memory:') # database=db_path

    # Словарь с названиями таблиц и путями к данным
    datasets = {
        'transactions': data_path / 'transactions_train.parquet',
        'articles': data_path / 'articles.parquet',
        'customers': data_path / 'customers.parquet'
    }

    # Загружаем данные в БД
    for name, path in datasets.items():

        query_start = f"CREATE OR REPLACE TABLE {name} AS "
        query_mid = "SELECT * "
        query_end = f"FROM read_parquet('{path.as_posix()}') "

        if name == 'transactions':
            # В таблице transactions меняем тип t_dat на DATE
            query_mid = "SELECT CAST(t_dat AS DATE) AS t_dat, * EXCLUDE (t_dat)"
    
        con.execute(f"""
            {query_start}{query_mid}
            {query_end}
        """)

    return con