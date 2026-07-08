import numpy as np
import torch

from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------- #
#                            Вспомогательные классы                            #
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Mapping:
    customer_id2index: dict
    customer_index2id: dict
    article_id2index: dict
    article_index2id: dict

@dataclass
class FeatureData:
    # -------------- Поля -------------- #
    # Данные
    numeric: np.ndarray
    categorical: np.ndarray
    # Названия столбцов
    numeric_columns: list[str]
    categorical_columns: dict[str: LabelEncoder]

    # ------------- Методы ------------- #
    # Кол-во числовых признаков
    def num_dim(self): 
        return len(self.numeric_columns)
    
    # Кол-во категорий в каждой категориальной переменной
    def cat_sizes(self):
        return [
            x['n_classes'] 
            for x in self.categorical_columns.values()
        ]

@dataclass
class PairData:
    customer_index: np.ndarray
    article_index: np.ndarray

@dataclass
class AggregateData:
    customers: FeatureData
    articles: FeatureData
    pairs: PairData
    mapping: Mapping

class TowerInfo:
    def __init__(self, df_aggr: AggregateData):
        self.customer_num_dim = df_aggr.customers.num_dim()
        self.article_num_dim = df_aggr.articles.num_dim()
        self.customer_cat_sizes = df_aggr.customers.cat_sizes()
        self.article_cat_sizes = df_aggr.articles.cat_sizes()


# ---------------------------------------------------------------------------- #
#                              Датасет для PyTorch                             #
# ---------------------------------------------------------------------------- #
class TwoTowerDataset(Dataset):

    # ------------------------------- Инициализация ------------------------------ #
    def __init__(self, tt_data: AggregateData):
        # Покупатели
        self.customer_num = torch.from_numpy(tt_data.customers.numeric)
        self.customer_cat = torch.from_numpy(tt_data.customers.categorical)

        # Товары
        self.article_num = torch.from_numpy(tt_data.articles.numeric)
        self.article_cat = torch.from_numpy(tt_data.articles.categorical)

        # Пары
        self.pair_customer_id = torch.from_numpy(tt_data.pairs.customer_index)
        self.pair_article_id = torch.from_numpy(tt_data.pairs.article_index)


    # -------------------------- Магический метод: длина ------------------------- #
    def __len__(self):
        return len(self.pair_customer_id)
    

    # ------------------ Магический метод: обращение по индексу ------------------ #
    def __getitem__(self, index):

        # Выделение индексов покупателя и товары из пары
        index_customer = self.pair_customer_id[index] - 1
        index_article = self.pair_article_id[index] - 1

        # Тензоры с данными о фичах
        customer_num = self.customer_num[index_customer]
        customer_cat = self.customer_cat[index_customer]
        article_num = self.article_num[index_article]
        article_cat = self.article_cat[index_article]

        return {
            # Покупатели
            'customer_idx': index_customer,
            'customer_num': customer_num,
            'customer_cat': customer_cat,
            
            # Товары
            'article_idx': index_article,
            'article_num': article_num,
            'article_cat': article_cat
        }
    