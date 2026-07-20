import torch

from scripts.two_tower.class_towers import TwoTower
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable

def tt_model_candidates(
        tt_model: TwoTower,
        data_loader_customer: DataLoader,
        data_loader_article: DataLoader,
        k: int
    ):
    """
    Генерация кандидатов на основе Two Tower
    """

    # Построение эмбеддингов покупателей и товаров
    customer_embeddings = tt_encode(
        tt_model, 
        data_loader_customer, 
        encode_customer=True
    )
    article_embeddings = tt_encode(
        tt_model, 
        data_loader_article, 
        encode_customer=False
    )

    # Топ-k товаров-кандидатов для пользователей
    tt_candidates_idx, tt_candidates_val = retrieve_candidates(
        customer_embeddings, 
        article_embeddings, 
        data_loader_customer, 
        tt_model, 
        k=k
    )

    return tt_candidates_idx, tt_candidates_val




# ---------------------------------------------------------------------------- #
#                            Вспомогательные функции                           #
# ---------------------------------------------------------------------------- #
@torch.inference_mode()
def tt_encode(
        tt_model: TwoTower,
        data_loader: DataLoader,
        encode_customer: bool
    ) -> torch.Tensor:
    """
    Экодинг объектов на основе оцененной нейросети Two Tower
    """

    # Перевод модели в режим инференса
    tt_model.eval()

    # Построение эмбеддингов
    if encode_customer: # для покупателей
        
        embeddings = batch_encoder(
            tt_model.encode_customers,
            data_loader,
            'customer'
        )
    
    else: # для товаров
        
        embeddings = batch_encoder(
            tt_model.encode_articles,
            data_loader,
            'article'
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

    # Проверка подстановки
    if tower not in ['customer', 'article']:
        raise ValueError('tower has to be either "customer" or "article"')
    
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


def retrieve_candidates(
        customer_embeddings: torch.Tensor, 
        article_embeddings: torch.Tensor,
        data_loader_customer: DataLoader,
        tt_model: TwoTower,
        k: int
    ):

    """
    Топ-k товаров-кандидатов для пользователей
    """

    # Предопределение
    tt_candidates_idx = [] # индексы наиболее релевантных товаров
    tt_candidates_val = [] # зачнения схожестей

    # Отбор товаров-кандидатов по батчам
    for batch in tqdm(data_loader_customer, desc=f'Top-{k} by similarity'):

        # Извлечение батча покупателей
        index = batch['customer_idx'] - 1
        customer_batch = customer_embeddings[index]

        # Скалярные произведения на батче
        similarities = tt_model.similarity(customer_batch, article_embeddings)

        # Извлечение k наиболее близких объектов
        top_k = torch.topk(similarities, k=k, dim=1)
        tt_candidates_idx.append(top_k.indices) # индексы
        tt_candidates_val.append(top_k.values) # схожести

    # Объединение в тензоры
    tt_candidates_idx = torch.cat(tt_candidates_idx, dim=0)
    tt_candidates_val = torch.cat(tt_candidates_val, dim=0)

    return tt_candidates_idx, tt_candidates_val