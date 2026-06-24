import _duckdb
from fastapi import HTTPException
from typing import List

def get_recommendations_batch(
        con: _duckdb.DuckDBPyConnection, 
        customer_ids: List[str],
        k: int = 12
    ):

    # Ни один пользователь не подан => ошибка
    if not customer_ids:
        raise HTTPException(status_code=400, detail='empty customer_ids')
    
    # Запрос к БД: рекомендации для пользователя
    data = con.execute(f"""
        WITH ranked AS (
            SELECT
                r.customer_id,
                ROW_NUMBER() OVER (
                    PARTITION BY r.customer_id
                    ORDER BY r.score DESC
                ) AS rating,
                r.article_id,
                a.product_type_name,
                a.graphical_appearance_name,
                a.detail_desc
            FROM recommendations r
            LEFT JOIN articles a ON a.article_id = r.article_id
            WHERE r.customer_id = ANY({customer_ids})
            ORDER BY r.score DESC
        )

        SELECT *
        FROM ranked
        WHERE rating <= {k}
    """).df()

    # Ни один пользователь не найден => ошибка
    if data.empty:
        raise HTTPException(status_code=404, detail='no user found')
    
    result = (data
        .groupby('customer_id')
        .apply(lambda x: {
            'customer_id': x.name, 
            'recommendations': x.to_dict(orient='records')
        })
        .to_list()
    )

    return result