from scripts.generate_features_article import generate_features_article
from scripts.generate_features_customer import generate_features_customer
from scripts.row_counts import row_counts

def generate_tt_dfs_features(
        con, 
        logger, 
        split_dates: dict
    ):

    # Фичи: покупатели
    logger.info('1. Create customer features')
    generate_features_customer(con, split_dates)
    row_counts(con, logger, 'customer_features')

    # Фичи: товары
    logger.info('2. Create article features')
    generate_features_article(con, split_dates)
    row_counts(con, logger, 'article_features')
