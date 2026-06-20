import pandas as pd

from catboost import Pool

def generate_pool(df):

    # ------------------------------ Правки к данным ----------------------------- #
    # Удаление даты
    df = df.drop(
        ['article_first_purchase_date', 'article_last_purchase_date'], 
        axis=1
    )

    # Указание cat_features
    cat_cols = [
        'club_member_status',
        'fashion_news_frequency',
    ]


    # ------------------------------- Создание Pool ------------------------------ #
    # Категории переменных
    i = df['customer_id'] # идентификатор покупателя
    y = df['target'] # таргет
    X = df.drop(['customer_id', 'article_id', 'target'], axis=1) # фичи
    
    # Объявление Pool
    pool = Pool(
        data=X,
        label=y,
        group_id=i,
        cat_features=cat_cols
    )

    return pool