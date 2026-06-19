import numpy as np

def ap_at_k(actual, predicted, k=12):

    predicted = predicted[:k] # топ-K рекомендаций
    hits = 0.0 # кол-во купленных товаров из рекомендованных
    score = 0.0 # суммарный скор

    for i, p in enumerate(predicted, start=1):
        if p in actual: # товар из рекомендаций куплен
            hits += 1
            score += hits / i

    if len(actual) == 0:
        return 0.0
    
    return score / min(len(actual), k)

    