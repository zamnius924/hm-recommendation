import numpy as np

from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder

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