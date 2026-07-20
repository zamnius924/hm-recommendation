# %%  Импорт библиотек
import json
import torch

from scripts.two_tower.class_loss import SymmetricCrossEntropyLoss
from scripts.two_tower.class_towers import TwoTower
from scripts.two_tower.generate_tt_datasets import generate_tt_train
from scripts.utils.paths import DATA_PROCESSED_DIR, MODELS_DIR
from scripts.two_tower.tt_model_train import tt_model_train
from torch import optim
from torch.utils.data import DataLoader

# %% Загрузка параметров
# Временное разделение на train-, valid- и test-выборки
with open(file=DATA_PROCESSED_DIR / 'split_dates.json', mode='r') as file:
    dates = json.load(file)

# %% Создание данных для Two-tower model
tt_dataset_train, tt_info_train = generate_tt_train(dates['train'])
tt_dataset_test, tt_info_test = generate_tt_train(dates['test'])

# %% Инициализация модели
# Экземпляр сети
tt_model = TwoTower(
    num_hidden_dims_customer=[512, 256, 128],
    num_hidden_dims_article=[512, 256, 128],
    emb_dims_customer=[8, 8], 
    emb_dims_article=[],
    tt_info=tt_info_train,
    temperature=0.03,
    drop_prob=0.1
)

# Функция потерь
criterion = SymmetricCrossEntropyLoss()

# Даталоадеры
data_loader_train = DataLoader(
    tt_dataset_train,
    batch_size=512,
    shuffle=True,
    num_workers=0
)

data_loader_test = DataLoader(
    tt_dataset_test,
    batch_size=512,
    shuffle=False,
    num_workers=0
)

# Оптимизатор
optimizer = optim.Adam(tt_model.parameters(), lr=1e-4)

# Кол-во эпох
num_epochs = 20

# %% Обучение модели
tt_model_train(
    tt_model,
    data_loader_train,
    data_loader_test,
    optimizer,
    criterion,
    num_epochs
)

# %% Сохранение модели
torch.save(obj=tt_model, f=MODELS_DIR / 'tt_model.pt')
