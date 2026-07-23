import _duckdb

from pathlib import Path

def save_db(
        con: _duckdb.DuckDBPyConnection,
        file: str,
        path_dir: Path
    ):

    # Путь к файлу
    path = (path_dir / file).as_posix()

    # Сохранение таблицы в файл
    con.execute(f"""
        COPY candidates TO '{path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)