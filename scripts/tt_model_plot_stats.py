import matplotlib.pyplot as plt
import numpy as np

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