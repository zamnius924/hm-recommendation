import pandas as pd

def build_mapping(*dfs):

    # Все товары и покупатели
    customer_series = []
    article_series = []

    for df in dfs:
        # Добавляем покупателей, если они есть
        if 'customer_id' in df.columns:
            customer_series.append(df['customer_id'])
        
        # Добавляем товары, если они есть
        if 'article_id' in df.columns:
            article_series.append(df['article_id'])

    # Все уникальные покупатели и товары
    all_customers = pd.concat(customer_series).unique()
    all_articles = pd.concat(article_series).unique()

    # Словари для отображения индетификаторов в индексы
    customer_id2index = {
        customer_id: i 
        for i, customer_id in enumerate(all_customers)
    }
    article_id2index = {
        article_id: i 
        for i, article_id in enumerate(all_articles)
    }

    # Словари для отображения индексов в идентификаторы (обратные словари)
    customer_index2id = {
        i: customer_id
        for customer_id, i in customer_id2index.items()
    }
    article_index2id = {
        i: article_id
        for article_id, i in article_id2index.items()
    }

    return {
        'customer_id2index': customer_id2index,
        'article_id2index': article_id2index,
        'customer_index2id': customer_index2id,
        'article_index2id': article_index2id
    }
