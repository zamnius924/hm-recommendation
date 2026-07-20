from scripts.utils.aggregate_tt_dfs import aggregate_tt_dfs
from scripts.data.build_mapping import build_mapping
from scripts.two_tower.class_tt_data import TwoTowerDataset, TowerInfo
from scripts.two_tower.generate_tt_dfs import generate_tt_dfs
from scripts.data.load_db import load_db

def generate_tt_train(
        split_dates: dict
    ) -> tuple[TwoTowerDataset, TowerInfo]:

    # --------------- Создание датафреймов с фичами и индексами пар -------------- #
    # Подключение к БД
    con = load_db()

    # Датафреймы с фичами и индексами пар
    df_customer, df_article, df_pairs = generate_tt_dfs(con, 
                                                        split_dates, 
                                                        return_df=True,
                                                        pairs=True)

    # Отключение от БД
    con.close()


    # ----------------- Форматирование данных для Two-tower model ---------------- #
    # Создание мэппинга
    mapping = build_mapping(df_customer, df_article, df_pairs)

    # Вспомогательные данные
    aggr_data = aggregate_tt_dfs(df_customer, 
                                 df_article, 
                                 df_pairs, 
                                 mapping,
                                 pairs=True)

    # Данные для Two-tower model
    tt_info = TowerInfo(aggr_data)
    tt_dataset = TwoTowerDataset(aggr_data)

    return tt_dataset, tt_info