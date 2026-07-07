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
class CustomerData:
    # Данные
    numeric: np.ndarray
    categorical: np.ndarray
    # Названия столбцов
    numeric_columns: list[str]
    categorical_columns: dict[str: LabelEncoder]

@dataclass
class ArticleData:
    # Данные
    numeric: np.ndarray
    categorical: np.ndarray
    # Названия столбцов
    numeric_columns: list[str]
    categorical_columns: dict[str: LabelEncoder]

@dataclass
class PairData:
    customer_index: np.ndarray
    article_index: np.ndarray

@dataclass
class TwoTowerData:
    customers: CustomerData
    articles: ArticleData
    pairs: PairData
    mapping: Mapping


# ---------------------------------------------------------------------------- #
#                              Датасет для PyTorch                             #
# ---------------------------------------------------------------------------- #
class TwoTowerDataset(Dataset):

    # ------------------------------- Инициализация ------------------------------ #
    def __init__(self, tt_data: TwoTowerData):
        # Покупатели
        self.customer_num = tt_data.customers.numeric
        self.customer_cat = tt_data.customers.categorical

        # Товары
        self.article_num = tt_data.articles.numeric
        self.article_cat = tt_data.articles.categorical

        # Пары
        self.pair_customer_id = tt_data.pairs.customer_index
        self.pair_article_id = tt_data.pairs.article_index


    # -------------------------- Магический метод: длина ------------------------- #
    def __len__(self):
        return len(self.pair_customer_id)
    

    # ------------------ Магический метод: обращение по индексу ------------------ #
    def __getitem__(self, index):

        # Выделение индексов покупателя и товары из пары
        index_customer = self.pair_customer_id[index] - 1
        index_article = self.pair_article_id[index] - 1

        # Тензоры с данными о фичах
        customer_num = torch.tensor(self.customer_num[index_customer])
        customer_cat = torch.tensor(self.customer_cat[index_customer])
        article_num = torch.tensor(self.article_num[index_article])
        article_cat = torch.tensor(self.article_cat[index_article])

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
    