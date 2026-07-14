import torch.nn as nn

from IPython.display import clear_output
from scripts.class_towers import TwoTower
from scripts.tt_model_inference import tt_model_inference
from scripts.tt_model_plot_stats import tt_model_plot_stats
from scripts.tt_model_training_step import tt_model_training_step
from torch import optim
from torch.utils.data import DataLoader

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
        train_loss = tt_model_training_step(
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

        # Визуализация
        tt_model_plot_stats(
            train_loss_history,
            test_loss_history,
            'Two-Tower model'
        )