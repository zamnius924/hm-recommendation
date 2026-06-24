from app.services.get_con import get_con
from app.services.get_customers import get_customers
from app.services.get_recommendations import get_recommendations
from app.services.get_recommendations_batch import get_recommendations_batch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List


# ------------------- Инициализация и завершение приложения ------------------ #
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Логика запуска
    app.state.con = get_con() # инициализация БД

    yield

    # Логика завершения
    app.state.con.close() # отключение БД


# ----------------------------- Валидация данных ----------------------------- #
# Ответ на запрос о рекомендациях
class ArticleRecommendation(BaseModel):
    rating: int
    article_id: str
    product_type_name: str
    graphical_appearance_name: str
    detail_desc: str

class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[ArticleRecommendation]

# Тело запроса recommendations_batch
class RecommendationBatchRequest(BaseModel):
    customer_ids: List[str]
    k: int = 12


# ----------------------- Инициализация веб-приложения ----------------------- #
app = FastAPI(
    title='H&M Recommendation API',
    version='1.0.0',
    lifespan=lifespan
)


# ------------------------------- API endpoints ------------------------------ #
# Корневой эндпоинт
@app.get('/')
def root():
    return {'status': 'ok'}

# Список индексов покупателей на основе их порядковых номеров
@app.get('/customers', response_model=List[str])
def customers(
        request: Request,
        start: int = 1,
        end: int = 10
    ) -> List[str]:
    
    con = request.app.state.con
    return get_customers(con, start, end)

# Рекомендации для одного пользователя
@app.get('/recommendations/{customer_id}', response_model=RecommendationResponse)
def recommendations(
        request: Request,
        customer_id: str, 
        k: int = 12,
    ) -> RecommendationResponse:
    
    con = request.app.state.con
    return get_recommendations(con, customer_id, k)

# Рекомендации для списка пользователей
@app.post('/recommendations_batch', response_model=List[RecommendationResponse])
def recommendations_batch(
        request: Request,
        payload: RecommendationBatchRequest
    ) -> List[RecommendationResponse]: 
    
    con = request.app.state.con
    return get_recommendations_batch(con, payload.customer_ids, payload.k)