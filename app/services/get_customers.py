import _duckdb
from fastapi import HTTPException

def get_customers(
        con: _duckdb.DuckDBPyConnection, 
        start: int, 
        end: int
    ):

    # Обработка ошибок:
    if start < 1:
        raise HTTPException(
            status_code=400,
            detail="start must be positive"
        )
    if end < start:
        raise HTTPException(
            status_code=400,
            detail="end must be greater than or equal to start"
        )

    # Запрос к БД: идентификаторы пользователей по позиционным индексам
    data = con.execute(f"""
        WITH customers AS (
            SELECT  
                customer_id,
                ROW_NUMBER() OVER (ORDER BY customer_id) AS row_num
            FROM (SELECT DISTINCT customer_id FROM recommendations)
            ORDER BY customer_id
        )
        
        SELECT customer_id
        FROM customers
        WHERE row_num BETWEEN {start} AND {end}
        ORDER BY row_num
    """).df()

    return data.customer_id.to_list()
