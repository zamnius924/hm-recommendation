import json
import pandas as pd

from pathlib import Path
from scripts.utils.paths import MODELS_PROD_CONFIG_DIR

def load_window_config(data_path: Path = MODELS_PROD_CONFIG_DIR):

    # Длины окон для формирования таргета и фичей
    with open(file=data_path / 'window_config.json', mode='r') as file:
        window_length = json.load(file)

    # Преобразование строковых значений в timedelta
    for key in window_length.keys():
        window_length[key] = pd.to_timedelta(window_length[key])
    
    return window_length