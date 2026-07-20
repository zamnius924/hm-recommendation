import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from IPython.display import clear_output
from scripts.two_tower.class_towers import TwoTower
from scripts.two_tower.tt_model_inference import tt_model_inference
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def tt_model_train(
        tt_model: TwoTower,
        data_loader_train: DataLoader,
        data_loader_test: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int
    ):

    # История обучения (лоссы на обучающей и тестовой выборках)
    train_loss_history, test_loss_history = [], []

    # Обучение модели
    for epoch in range(num_epochs):

        # Обучение
        train_loss = tt_model_train_epoch(
            tt_model,
            data_loader_train,
            optimizer,
            criterion
        )

        # Инференс
        test_loss = tt_model_inference(
            tt_model,
            data_loader_test,
            criterion
        )

        # Добавление результатов
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        clear_output()

        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print(f'Train loss: {train_loss}')
        print(f'Test loss: {test_loss}')

        # Визуализация
        tt_model_plot_stats(
            train_loss_history,
            test_loss_history,
            'Two-Tower model'
        )




# ---------------------------------------------------------------------------- #
#                            Вспомогательные функции                           #
# ---------------------------------------------------------------------------- #
def tt_model_train_epoch(
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

    return total_loss


def tt_model_plot_stats(
        train_loss: list,
        test_loss: list,
        title: str
    ):

    # Номера эпох
    epochs = range(1, len(train_loss) + 1)

    # Инициализация полотна
    plt.figure(figsize=(12, 8))

    # Построение графиков
    plt.plot(epochs, train_loss, marker='o', label='Train loss')
    plt.plot(epochs, test_loss, marker='o', label='Test loss')
    
    # Подписи
    plt.title(title) # название
    plt.xlabel('Epoch') # ось X
    plt.ylabel('Loss') # ось Y

    # Дополнительные настройки
    plt.xticks(epochs) # целочисленные метки эпох
    plt.legend() # легенда
    plt.grid() # сетка

    plt.show()


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

    return total_loss