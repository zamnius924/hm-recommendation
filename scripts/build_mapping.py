import pandas as pd

def build_mapping(*dfs):

    # Все уникальные покупатели и товары
    all_customers = pd.concat([df.customer_id for df in dfs]).unique()
    all_articles = pd.concat([df.article_id for df in dfs]).unique()

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
