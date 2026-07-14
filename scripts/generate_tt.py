from scripts.aggregate_tt_dfs import aggregate_tt_dfs
from scripts.build_mapping import build_mapping
from scripts.class_tt_data import TwoTowerDataset, TowerInfo
from scripts.generate_tt_dfs import generate_tt_dfs
from scripts.load_db import load_db

def generate_tt(split_dates: dict):

    # --------------- Создание датафреймов с фичами и индексами пар -------------- #
    # Подключение к БД
    con = load_db()

    # Датафреймы с фичами и индексами пар
    df_customer, df_article, df_pairs = generate_tt_dfs(con, 
                                                        split_dates, 
                                                        return_df=True)

    # Отключение от БД
    con.close()


    # ----------------- Форматирование данных для Two-tower model ---------------- #
    # Создание мэппинга
    mapping = build_mapping(df_customer, df_article, df_pairs)

    # Вспомогательные данные
    df_aggr = aggregate_tt_dfs(df_customer, df_article, df_pairs, mapping)

    # Данные для Two-tower model
    tt_info = TowerInfo(df_aggr)
    tt_dataset = TwoTowerDataset(df_aggr)

    return tt_dataset, tt_info