def load_tt_dfs_pairs(con):

    # Создание датафреймов pandas
    df_pairs = con.execute("""
        SELECT * FROM pairs ORDER BY customer_id, article_id
    """).df()

    return df_pairs