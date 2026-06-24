from pydantic import BaseModel, Field
from typing import List

# ----------------------------- Валидация данных ----------------------------- #
# Ответ на запрос о рекомендациях
class ArticleRecommendation(BaseModel): # одна рекомендация
    rating: int
    article_id: str
    product_type_name: str
    graphical_appearance_name: str
    detail_desc: str

class RecommendationResponse(BaseModel): # набор рекомендаций покупателю
    customer_id: str
    recommendations: List[ArticleRecommendation]

# Тело запроса recommendations_batch
class RecommendationBatchRequest(BaseModel):
    customer_ids: List[str]
    k: int = Field(default=12, ge=1, le=100)