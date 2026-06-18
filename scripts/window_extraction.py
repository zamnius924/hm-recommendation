def window_extraction(con, dates: dict, sample: str):

    # Выделение feature-window
    df_feature = con.execute(f"""
        SELECT *
        FROM transactions
        WHERE t_dat BETWEEN 
            '{dates[sample]['feature_window_start']}' AND 
            '{dates[sample]['feature_window_end']}'
    """).df()

    # Выделение target-window
    df_target = con.execute(f"""
        SELECT *
        FROM transactions
        WHERE t_dat BETWEEN 
            '{dates[sample]['target_window_start']}' AND 
            '{dates[sample]['target_window_end']}'
    """).df()

    print(f'Кол-во наблюдений на feature_window: {len(df_feature)}')
    print(f'Кол-во наблюдений на target_window: {len(df_target)}')

    return df_feature, df_target