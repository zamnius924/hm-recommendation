from torch.utils.data import DataLoader, Dataset

def generate_loader(
        dataset: Dataset,
        mode: str,
        num_workers: int = 0,
        batch_size: int = 512
    ) -> DataLoader:

    # Проверка входных параметров
    if mode not in ['train', 'eval']:
        raise ValueError('mode has to be either "train" or "eval"')

    # Инициализация даталоадеров для покупателей и товаров
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers
    )

    return data_loader
