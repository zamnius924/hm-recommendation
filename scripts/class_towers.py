import torch
import torch.nn as nn

# ---------------------------------------------------------------------------- #
#                                     Башня                                    #
# ---------------------------------------------------------------------------- #
class Tower(nn.Module):

    # ---------------------------- Инициализация башни --------------------------- #
    def __init__(
            self,
            num_input_dim: int, # кол-во входных числовых параметров
            num_hidden_dim: int, # кол-во скрытых слоев MLP
            cat_sizes: list[int], # кол-во категорий в каждой категориальной переименной
            emb_dims: list[int], # размеры эмбеддингов
        ):

        # Проверка входных параметров
        if len(cat_sizes) != len(emb_dims):
            raise ValueError(
                'cat_sizes and emb_dims must have the same length'
            )

        # Наследование от nn.Module
        super().__init__()

        # Вспомогательные параметры
        self.use_emb = len(cat_sizes) > 0 # наличие категориальных признаков
        self.cat_dim = len(cat_sizes) # кол-во категориальных признаков

        # Полносвязные слои
        input_dim = num_input_dim + sum(emb_dims) # размерность с учетом эмбеддингов

        self.linear_1 = nn.Linear(in_features=input_dim, 
                                  out_features=num_hidden_dim)
        self.linear_2 = nn.Linear(in_features=num_hidden_dim, 
                                  out_features=num_hidden_dim)
        
        # Слои-эмбединги
        if self.use_emb:
            self.emb = nn.ModuleList([
                nn.Embedding(
                    num_embeddings=n_classes, 
                    embedding_dim=embedding_dim
                )
                for embedding_dim, n_classes in zip(emb_dims, cat_sizes)
            ])
        else:
            self.emb = None

        # Батч-нормализация
        self.batch_norm = nn.BatchNorm1d(num_features=num_input_dim)
        
        # Активация
        self.activation = nn.ReLU()


    # ----------------------------- Архитектура сети ----------------------------- #
    def forward(self, x_num, x_cat):

        # Батч-нормализация числовых признаков
        x_num = self.batch_norm(x_num)

        # Обработка категориальных признаков
        if self.use_emb:
            # Эмбеддинг категориальных признаков
            z_cat = torch.cat(
                [emb(x_cat[:,i]) for i, emb in enumerate(self.emb)],
                dim=1
            )
            # Объединение числовых признаков и эмбеддингов
            x = torch.cat([x_num, z_cat], dim=1)
        else:
            x = x_num

        # Подача данных в MLP
        z = self.linear_1(x)
        z = self.activation(z)
        z = self.linear_2(z)

        return z


# ---------------------------------------------------------------------------- #
#                                   Две башни                                  #
# ---------------------------------------------------------------------------- #
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

