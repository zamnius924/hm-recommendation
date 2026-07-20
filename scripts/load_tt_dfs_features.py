def load_tt_dfs_features(con):

    # Создание датафреймов pandas
    df_customer = con.execute("""
        SELECT * FROM customer_features ORDER BY customer_id
    """).df()
    df_article = con.execute("""
        SELECT * FROM article_features ORDER BY article_id
    """).df()

    return df_customer, df_article