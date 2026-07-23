import pandas as pd
import torch

from scripts.two_tower.class_towers import TwoTower
from scripts.two_tower.class_tt_data import CandidateDataset, Mapping
from scripts.two_tower.generate_loader import generate_loader
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable

def generate_tt_candidates(
        tt_model: TwoTower,
        candidates: CandidateDataset,
        k: int
    ):
    """
    Генерация кандидатов на основе результатов Two Tower
    """

    # Инициализация даталоадеров для покупателей и товаров
    data_loader_customer = generate_loader(candidates.customer, 
                                           mode='eval')
    data_loader_article = generate_loader(candidates.article, 
                                          mode='eval')

    # Построение эмбеддингов покупателей и товаров
    customer_embeddings = tt_encode(
        tt_model, 
        data_loader_customer, 
        'customer'
    )
    article_embeddings = tt_encode(
        tt_model, 
        data_loader_article, 
        'article'
    )

    # Топ-k товаров-кандидатов для пользователей
    candidate_df = tt_ranking(
        customer_embeddings, 
        article_embeddings, 
        data_loader_customer, 
        tt_model, 
        candidates.mapping,
        k=k
    )

    return candidate_df




# ---------------------------------------------------------------------------- #
#                            Вспомогательные функции                           #
# ---------------------------------------------------------------------------- #
# -------------------------------- 1) Энкодинг ------------------------------- #
@torch.inference_mode()
def tt_encode(
        tt_model: TwoTower,
        data_loader: DataLoader,
        tower: str
    ) -> torch.Tensor:
    """
    Экодинг объектов на основе оцененной нейросети Two Tower
    """

    # Перевод модели в режим инференса
    tt_model.eval()

    # Проверка подстановки
    if tower not in {'customer', 'article'}:
        raise ValueError('tower has to be either "customer" or "article"')

    # Выбор энкодера
    if tower == 'customer': # для покупателей
        encoder = tt_model.encode_customers    
    else: # для товаров
        encoder = tt_model.encode_articles

    # Построение эмбеддингов    
    embeddings = batch_encoder(
        encoder,
        data_loader,
        tower
    )

    return torch.cat(embeddings, dim=0)


def batch_encoder(
        encoder: Callable[
            [torch.Tensor, torch.Tensor], # типы аргументов функции
            torch.Tensor # тип возвращаемого объекта
        ],
        data_loader: DataLoader,
        tower: str
    ) -> list:
    """
    Экодинг по батчам
    """
    
    # Предопределение
    embeddings = []

    for batch in tqdm(data_loader, desc=f'Encoding: {tower}'):

        # Энкодинг
        emb = encoder(
            batch[f'{tower}_num'],
            batch[f'{tower}_cat']
        )

        # Добавление энкодингов на батче
        embeddings.append(emb)

    return embeddings


# --------------------- 2) Датафрейм с парами кандидатов --------------------- #
def tt_ranking(
        customer_embeddings: torch.Tensor, 
        article_embeddings: torch.Tensor,
        data_loader_customer: DataLoader,
        tt_model: TwoTower,
        mapping: Mapping,
        k: int
    ) -> pd.DataFrame:

    """
    Генерация кандидатов по батчам
    """

    # Предопределение
    candidate_df = []

    # Отбор товаров-кандидатов по батчам
    for batch in tqdm(data_loader_customer, desc=f'Top-{k} by similarity'):

        # Топ кандидаты
        top_k = top_candidates(
            customer_embeddings,
            article_embeddings,
            batch,
            tt_model,
            k
        )

        # Индексы (приведены к векторам)
        customer_idx = batch['customer_idx'].repeat_interleave(k) # повтор k раз
        article_idx = top_k.indices.reshape(-1)

        # Перевод индексов в идентификаторы
        customer_id = [mapping.customer_index2id[i.item()] for i in customer_idx]
        article_id = [mapping.article_index2id[i.item()] for i in article_idx]
        
        # Скоры (приведены к векторам)
        scores = top_k.values.reshape(-1)

        # Создаем вспомогательный датафрейм на батче
        candidate_batch = pd.DataFrame({
            'customer_id': customer_id,
            'article_id': article_id,
            'scores': scores
        })

        # Сохраняем датафрейм батча в список
        candidate_df.append(candidate_batch)

    # Объединение в тензоры
    candidate_df = pd.concat(candidate_df, axis=0)

    return candidate_df


def top_candidates(
        customer_embeddings: torch.Tensor, 
        article_embeddings: torch.Tensor,
        batch: dict,
        tt_model: TwoTower,
        k: int
    ):

    """
    Топ-k товаров-кандидатов для пользователей на батче
    """

    # Извлечение батча покупателей
    index = batch['customer_idx']
    customer_batch = customer_embeddings[index]

    # Скалярные произведения на батче
    similarities = tt_model.similarity(customer_batch, article_embeddings)

    # Извлечение k наиболее близких объектов
    top_k = torch.topk(similarities, k=k, dim=1)
    
    return top_k