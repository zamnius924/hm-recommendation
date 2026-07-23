from pathlib import Path

# Объявление путей к директориям проекта
ROOT = Path(__file__).parent.parent.parent

# ----------------------------------- Data ----------------------------------- #
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'

DATA_RAW_DIR = DATA_DIR / 'raw'
DATA_PROCESSED_DIR = DATA_DIR / 'processed'
DATA_PROD_DIR = DATA_DIR / 'production'
DATA_RECOMMENDATIONS_DIR = DATA_DIR / 'recommendations'

DATA_PROCESSED_MOD1_DIR = DATA_PROCESSED_DIR / 'model_1'
DATA_PROCESSED_MOD2_DIR = DATA_PROCESSED_DIR / 'model_2'

# ---------------------------------- Models ---------------------------------- #
MODELS_CONFIG_DIR = MODELS_DIR / 'config'
MODELS_PROD_DIR = MODELS_DIR / 'production'
MODELS_PROD_CONFIG_DIR = MODELS_PROD_DIR / 'config'
