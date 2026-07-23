import pandas as pd

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Уменьшает типы данных для экономии места на диске и в памяти.
    - Целочисленные колонки: наименьший подходящий int
    - Вещественные колонки: float32
    """

    # Целые -> наименьший знаковый int, вмещающий диапазон
    for col in df.select_dtypes(include=['int64', 'int32', 'int16']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # Вещественные -> float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    return df
