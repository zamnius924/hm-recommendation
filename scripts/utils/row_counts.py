import _duckdb
import logging

def row_counts(
        con: _duckdb.DuckDBPyConnection, 
        logger: logging.Logger, 
        table: str
    ):

    # Счетчик строк в таблице
    n_rows = con.execute(f"""
        SELECT COUNT(*)
        FROM {table}
    """).fetchone()[0]

    # Размер таблицы в логи
    logger.info(f'{table}: %s rows', f'{n_rows:,}')