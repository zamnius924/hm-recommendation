import pandas as pd

from pathlib import Path
from scripts.data.load_db import load_db
from scripts.data.save_db import save_db
from scripts.features.generate_features import generate_features

def build_feature_table(
        split_dates: dict, 
        candidates: pd.DataFrame, 
        file: str, 
        path_dir: Path,
        target: bool = True
    ):

    # Инициализация БД
    con = load_db()

    # Генерация фичей для кандидатов
    con = generate_features(
        con,
        split_dates,
        candidates,
        target=target,
        return_con=True
    )

    # Сохранение таблицы
    save_db(con, file, path_dir=path_dir)

    # Отключение БД
    con.close()