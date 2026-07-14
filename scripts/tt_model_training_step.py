import torch.nn as nn

from scripts.class_towers import TwoTower
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def tt_model_training_step(
        tt_model: TwoTower,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> float:

    # Перевод модели в режим обучения
    tt_model.train()

    # Потери на эпохе
    total_loss = 0

    for batch in tqdm(data_loader, desc='Train'):

        # Обнуление градиента
        optimizer.zero_grad()

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

        # Backward pass – оценка производных и градиента функции потерь
        loss.backward()

        # Шаг градиентного спуска – новые значения параметров
        optimizer.step()

    # Усреднение потерь по батчам
    total_loss /= len(data_loader)

    print(f'Loss on epoch (training): {total_loss}')

    return total_loss