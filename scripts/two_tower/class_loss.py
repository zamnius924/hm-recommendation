import torch
import torch.nn as nn

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__() # наследование от nn.Module
        self.criterion = nn.CrossEntropyLoss(**kwargs) # односторонний лосс

    def forward(self, logits: torch.Tensor):

        # Таргет - положительный класс на диагонали
        target = torch.arange(logits.shape[0], device=logits.device)

        # Логиты для покупателей и товаров
        logits_customer = logits
        logits_article = logits.T

        # Лосс для задачи: рекомендовать каждому покупателю правильный товар
        loss_customer = self.criterion(logits_customer, target) 
        # Лосс для задачи: подобрать каждому товару "правильного" покупателя
        loss_article = self.criterion(logits_article, target)

        return (loss_customer + loss_article) / 2
