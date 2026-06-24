from app.get_con import get_con
#from app.get_customers import get_customers
from app.get_recommendations import get_recommendations
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
class ArticleRecommendation(BaseModel):
    rating: int
    article_id: str
    product_type_name: str
    graphical_appearance_name: str
    detail_desc: str

class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[ArticleRecommendation]


# ----------------------- Инициализация веб-приложения ----------------------- #
app = FastAPI(lifespan=lifespan)


# ------------------------------- API endpoints ------------------------------ #
@app.get('/')
def root():
    return {'status': 'ok'}

#@app.get('/customers')
#def customers(index: List[int]):
#    return get_customers(index)

@app.get('/recommendations/{customer_id}')
def recommendations(
        request: Request,
        customer_id: str, 
        k: int = 12,
    ) -> RecommendationResponse:
    
    con = request.app.state.con
    recs = get_recommendations(con, customer_id, k)

    return recs