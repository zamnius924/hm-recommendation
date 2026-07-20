import numpy as np

from scripts.evaluation.ap_at_k import ap_at_k

def map_at_k(df, y_hat, k=12):

    df['score'] = y_hat
    ap_scores = []

    for customer_id, df_customer in df.groupby('customer_id'):

        actual = ( # список купленных товаров
            df_customer
                .loc[df_customer['target']==1, 'article_id']
                .tolist()
        )
        predicted = ( # рекомендованные товары, отсортированные по score
            df_customer
                .sort_values('score', ascending=False)
                ['article_id']
                .tolist()
        )

        ap_scores.append(ap_at_k(actual, predicted, k))

    return np.mean(ap_scores)
