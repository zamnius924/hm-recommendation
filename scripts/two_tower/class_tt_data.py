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
    def __init__(
            self, 
            df_aggr: AggregateData
        ):

        self.customer_num_dim = df_aggr.customers.num_dim()
        self.article_num_dim = df_aggr.articles.num_dim()
        self.customer_cat_sizes = df_aggr.customers.cat_sizes()
        self.article_cat_sizes = df_aggr.articles.cat_sizes()




# ---------------------------------------------------------------------------- #
#                           Вспомогательные датасеты                           #
# ---------------------------------------------------------------------------- #
class FeatureDataset(Dataset):

    # ------------------------------- Инициализация ------------------------------ #
    def __init__(
            self, 
            numeric: np.ndarray, 
            categorical: np.ndarray
        ):
        self.numeric = torch.from_numpy(numeric.copy())
        self.categorical = torch.from_numpy(categorical.copy())

    # -------------------------- Магический метод: длина ------------------------- #
    def __len__(self):
        return len(self.numeric)
    
    # --------------------------- Вспомогательный метод -------------------------- #
    def get(self, index):
        return self.numeric[index], self.categorical[index]


class CustomerDataset(FeatureDataset):

    # ------------------------------- Инициализация ------------------------------ #
    def __init__(
            self,
            customer_data: FeatureData
        ):
        super().__init__(customer_data.numeric, customer_data.categorical)

    # ------------------ Магический метод: обращение по индексу ------------------ #
    def __getitem__(self, index):

        customer_num, customer_cat = self.get(index) # фичи

        return {
            'customer_idx': index,
            'customer_num': customer_num,
            'customer_cat': customer_cat
        }


class ArticleDataset(FeatureDataset):

    # ------------------------------- Инициализация ------------------------------ #
    def __init__(
            self,
            article_data: FeatureData
        ):
        super().__init__(article_data.numeric, article_data.categorical)

    # ------------------ Магический метод: обращение по индексу ------------------ #
    def __getitem__(self, index):
        
        article_num, article_cat = self.get(index) # фичи

        return {
            'article_idx': index,
            'article_num': article_num,
            'article_cat': article_cat
        }




# ---------------------------------------------------------------------------- #
#                             Датасет для обучения                             #
# ---------------------------------------------------------------------------- #
class TwoTowerDataset(Dataset):

    # ------------------------------- Инициализация ------------------------------ #
    def __init__(
            self, 
            tt_data: AggregateData
        ):
        # Покупатели
        self.customers = FeatureDataset(
            numeric=tt_data.customers.numeric,
            categorical=tt_data.customers.categorical
        )

        # Товары
        self.articles = FeatureDataset(
            numeric=tt_data.articles.numeric,
            categorical=tt_data.articles.categorical
        )

        # Пары
        self.pair_customer_id = torch.from_numpy(tt_data.pairs.customer_index.copy())
        self.pair_article_id = torch.from_numpy(tt_data.pairs.article_index.copy())


    # -------------------------- Магический метод: длина ------------------------- #
    def __len__(self):
        return len(self.pair_customer_id)
    

    # ------------------ Магический метод: обращение по индексу ------------------ #
    def __getitem__(self, index):

        # Выделение индексов покупателя и товары из пары
        index_customer = self.pair_customer_id[index] - 1
        index_article = self.pair_article_id[index] - 1

        # Тензоры с данными о фичах
        customer_num, customer_cat = self.customers.get(index_customer)
        article_num, article_cat = self.articles.get(index_article)

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
