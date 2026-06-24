import _duckdb
from fastapi import HTTPException

def get_recommendations(
        con: _duckdb.DuckDBPyConnection, 
        customer_id: str, 
        k: int
    ):

    # Запрос к БД: рекомендации для пользователя
    data = con.execute(f"""
        SELECT
            ROW_NUMBER() OVER (ORDER BY r.score DESC) AS rating,
            r.article_id,
            a.product_type_name,
            a.graphical_appearance_name,
            a.detail_desc
        FROM recommendations r
        LEFT JOIN articles a ON a.article_id = r.article_id
        WHERE r.customer_id = '{customer_id}'
        ORDER BY r.score DESC
        LIMIT {k}
    """).df()

    if data.empty:
        # Пользователь не найден => ошибка
        raise HTTPException(404, 'no user found')
    else:
        # Пользователь найден => возвращаем его рекомендации
        return {
            'customer_id': customer_id,
            'recommendations': data.to_dict(orient='records')
        }