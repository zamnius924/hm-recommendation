import matplotlib.pyplot as plt
import numpy as np

def tt_model_plot_stats(
        train_loss: list,
        test_loss: list,
        title: str
    ):

    # Инициализация полотна
    plt.figure(figsize=(16, 8))

    # Построение графиков
    plt.plot(train_loss, label='Train loss')
    plt.plot(test_loss, label='Test loss')
    
    # Дополнительные параметры
    plt.title(title) # название
    plt.legend() # легенда
    plt.grid() # сетка

    plt.show()