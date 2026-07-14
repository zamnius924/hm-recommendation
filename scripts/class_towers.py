import torch
import torch.nn as nn

from scripts.class_tt_data import TowerInfo

# ---------------------------------------------------------------------------- #
#                                     Башня                                    #
# ---------------------------------------------------------------------------- #
class Tower(nn.Module):

    # ---------------------------- Инициализация башни --------------------------- #
    def __init__(
            self,
            num_input_dim: int, # кол-во входных числовых параметров
            num_hidden_dim: int, # кол-во скрытых слоев MLP
            cat_sizes: list[int], # кол-во категорий в каждой категориальной переменной
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
        self.activation = nn.GELU()


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
    def __init__(
            self, 
            # Кол-во скрытых слоев MLP
            num_hidden_dim_customer: int,
            num_hidden_dim_article: int,
            # Размеры эмбеддингов
            emb_dims_customer: list[int], 
            emb_dims_article: list[int],
            # Параметры башен
            tt_info: TowerInfo,
            temperature: float
        ):

        # Наследование от nn.Module
        super().__init__()

        # Башня: покупатели
        self.customer = Tower(
            num_input_dim=tt_info.customer_num_dim,
            num_hidden_dim=num_hidden_dim_customer,
            cat_sizes=tt_info.customer_cat_sizes,
            emb_dims=emb_dims_customer
        )
        
        # Башня: товары
        self.article = Tower(
            num_input_dim=tt_info.article_num_dim,
            num_hidden_dim=num_hidden_dim_article,
            cat_sizes=tt_info.article_cat_sizes,
            emb_dims=emb_dims_article
        )

        # Дополнительные параметры
        self.temperature = temperature

    # ----------------------------- Архитектура сети ----------------------------- #
    def forward(
            self, 
            x_num_customer: torch.Tensor, 
            x_cat_customer: torch.Tensor, 
            x_num_article: torch.Tensor,
            x_cat_article: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:

        # Эмбеддинги покупателей и товаров
        u = self.customer(x_num_customer, x_cat_customer)
        v = self.article(x_num_article, x_cat_article)

        return u, v
    
    # --------------------------- Дополнительные методы -------------------------- #
    def similarity(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        # Нормализация эмбеддингов
        u_norm = nn.functional.normalize(u, dim=1, eps=1e-8)
        v_norm = nn.functional.normalize(v, dim=1, eps=1e-8)

        # Скалярное произведение нормализованных векторов => cosine similarity
        dot_prod = u_norm @ v_norm.T / self.temperature
        
        return dot_prod

