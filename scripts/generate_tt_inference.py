from scripts.aggregate_tt_dfs import aggregate_tt_dfs
from scripts.build_mapping import build_mapping
from scripts.class_tt_data import CustomerDataset, ArticleDataset
from scripts.generate_tt_dfs import generate_tt_dfs
from scripts.load_db import load_db

def generate_tt_inference(
        split_dates: dict
    ) -> tuple[CustomerDataset, ArticleDataset, dict]:

    # --------------- Создание датафреймов с фичами и индексами пар -------------- #
    # Подключение к БД
    con = load_db()

    # Датафреймы с фичами и индексами пар
    df_customer, df_article = generate_tt_dfs(con, 
                                              split_dates, 
                                              return_df=True,
                                              pairs=False)

    # Отключение от БД
    con.close()

    # ----------------- Форматирование данных для Two-tower model ---------------- #
    # Создание мэппинга
    mapping = build_mapping(df_customer, df_article)

    # Вспомогательные данные
    aggr_data = aggregate_tt_dfs(df_customer, 
                                 df_article, 
                                 None, 
                                 mapping,
                                 pairs=False)
    
    customer_dataset = CustomerDataset(aggr_data.customers)
    article_dataset = ArticleDataset(aggr_data.articles)

    return customer_dataset, article_dataset, mapping