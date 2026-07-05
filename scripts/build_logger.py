import logging

def build_logger():

    # Конфигурации логгера
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    # Объявление логгера
    logger = logging.getLogger(__name__)

    return logger