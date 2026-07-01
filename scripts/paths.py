from pathlib import Path

# Объявление путей к директориям проекта
ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'

DATA_RAW_DIR = DATA_DIR / 'raw'
DATA_PROCESSED_DIR = DATA_DIR / 'processed'
DATA_PROD_DIR = DATA_DIR / 'production'
DATA_RECOMMENDATIONS_DIR = DATA_DIR / 'recommendations'

MODELS_CONFIG_DIR = MODELS_DIR / 'config'
MODELS_PROD_DIR = MODELS_DIR / 'production'
MODELS_PROD_CONFIG_DIR = MODELS_PROD_DIR / 'config'
