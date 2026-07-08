import torch
import torch.nn as nn

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__() # наследование от nn.Module
        self.criterion = nn.CrossEntropyLoss() # односторонний лосс

    def forward(self, logits: torch.Tensor):

        # Таргет - положительный класс на диагонали
        target = torch.arange(logits.shape[0], device=logits.device)

        # Логиты для покупателей и товаров
        logits_customer = logits
        logits_article = logits.T

        # Потери для покупателей и товаров
        loss_customer = self.criterion(logits_customer, target) # покупателю правильный товар
        loss_article = self.criterion(logits_article, target) # товару правильный покупатель

        return (loss_customer + loss_article) / 2
