import torch
import torch.nn as nn

class Tower(nn.Module):

    # ---------------------------- Инициализация башни --------------------------- #
    def __init__(self, input_dim, hidden_dim):
        super().__init__() # наследование от nn.Module

        # Слои
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        # Активация
        self.activation = nn.ReLU()

    # ----------------------------- Архитектура сети ----------------------------- #
    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        output = self.linear_2(x)

        return output


class TwoTower(nn.Module):

    # ------------------------- Инициализация двух башен ------------------------- #
    def __init__(self, input_dim_customer, input_dim_article, hidden_dim):
        super().__init__() # наследование от nn.Module

        # Башни
        self.customer = Tower(input_dim_customer, hidden_dim)
        self.article = Tower(input_dim_article, hidden_dim)

    # ----------------------------- Архитектура сети ----------------------------- #
    def forward(self, x_customer, x_article):
        u = self.customer(x_customer)
        v = self.article(x_article)

        score = torch.sum(u * v, dim=1)

        return score

