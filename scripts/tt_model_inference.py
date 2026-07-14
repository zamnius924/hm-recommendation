import torch
import torch.nn as nn

from scripts.class_towers import TwoTower
from torch.utils.data import DataLoader
from tqdm import tqdm

@torch.inference_mode()
def tt_model_inference(
        tt_model: TwoTower,
        data_loader: DataLoader,
        criterion: nn.Module
    ) -> float:

    # Перевод модели в режим инференса
    tt_model.eval()

    # Потери на эпохе
    total_loss = 0

    for batch in tqdm(data_loader, desc='Inference'):

        # Forward pass – эмбеддинги покупателей и товаров
        u, v = tt_model(
            batch['customer_num'], 
            batch['customer_cat'], 
            batch['article_num'], 
            batch['article_cat']
        )

        # Матрица скалярных произведений
        logits = tt_model.similarity(u, v)

        # Значение функционала потерь
        loss = criterion(logits)
        total_loss += loss.item()

    # Усреднение потерь по батчам
    total_loss /= len(data_loader)

    print(f'Loss on epoch (inference): {total_loss}')

    return total_loss