import pandas as pd

from tqdm import tqdm
from scripts.ltr.generate_pool import generate_pool

def generate_scores(con, model, batch_size):

    # Все кандидаты, для которых будут строиться рекомендации
    all_customers = con.execute("""
        SELECT DISTINCT customer_id 
        FROM als_candidates
    """).df()

    # Предопределение
    rec_list = [] # список рекомендаций для каждого батча
    
    for start in tqdm(range(0, len(all_customers), batch_size)):

        # ------------------------ Создание батча покупателей ------------------------ #
        # Конечный индекс покупателя
        end = start + batch_size

        # Батч покупателей
        batch_customers = pd.DataFrame({
            'customer_id': all_customers.customer_id[start:end]
        })

        # Подключение батча к БД
        con.register('batch_customers', batch_customers)

        # Сортировка пользователей по батчу
        df_batch = con.execute(f"""
            SELECT *
            FROM als_candidates
            WHERE customer_id IN (SELECT customer_id FROM batch_customers)
            ORDER BY customer_id, article_id
        """).df()

        # Отключение батча от БД
        con.unregister('batch_customers')


        # ----------------------------- Прогноз на батче ----------------------------- #
        # Создание пула
        pool_batch = generate_pool(df_batch, target=False)

        # Прогноз для покупателей
        pred_batch = model.predict(pool_batch)

        # Сохранение скора и удаление фичей
        df_batch = pd.DataFrame({
            'customer_id': df_batch['customer_id'],
            'article_id': df_batch['article_id'],
            'score': pred_batch
        })

        # Добавление результата в список
        rec_list.append(df_batch)

    # Соединение результатов в датафрейм
    df_rec = pd.concat(rec_list, ignore_index=True)

    return df_rec