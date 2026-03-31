"""
FastAPI-бэкенд для сервиса анализа комментариев

Запуск из корня репозитория:
    uvicorn src.api.main:app --reload

Swagger UI: http://localhost:8000/docs
"""

import sys
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import PRODUCT_TO_ID, run_pipeline
from src.db.queries import get_product_stats, get_products

ID_TO_PRODUCT = {v: k for k, v in PRODUCT_TO_ID.items()}

app = FastAPI(
    title="Анализ тем пользовательских комментариев",
    description="Пайплайн: BERTopic + LLM+RAG -> именованные темы из открытых комментариев.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    product_id: str
    date_from: date
    date_to: date

    @field_validator("date_to")
    @classmethod
    def date_range_valid(cls, date_to: date, values) -> date:
        date_from = values.data.get("date_from")
        if date_from and date_to < date_from:
            raise ValueError("date_to не может быть раньше date_from")
        return date_to


@app.get("/health", summary="Проверка доступности сервиса")
def health():
    return {"status": "ok"}


@app.get("/products", summary="Список продуктов из БД")
def products():
    """
    Возвращает список продуктов с их идентификаторами.
    Используется для выпадающего списка
    """
    names = get_products()
    return [
        {"id": PRODUCT_TO_ID[name], "name": name}
        for name in names
        if name in PRODUCT_TO_ID
    ]


@app.get("/products/{product_id}/stats", summary="Статистика по продукту")
def product_stats(product_id: str):
    """
    Возвращает диапазон доступных дат и количество комментариев.
    Используется для функции выбора дат
    """
    product_name = ID_TO_PRODUCT.get(product_id)
    if not product_name:
        raise HTTPException(
            status_code=404,
            detail=f"Продукт '{product_id}' не найден. "
            f"Доступные: {list(ID_TO_PRODUCT.keys())}",
        )
    stats = get_product_stats(product_name)
    return {"product_id": product_id, "name": product_name, **stats}


@app.post("/analyze", summary="Запустить анализ комментариев")
def analyze(req: AnalyzeRequest):
    """
    Принимает продукт и период, запускает полный пайплайн.
    В среднем выполняется за 2-3 минуты
    """
    product_name = ID_TO_PRODUCT.get(req.product_id)
    if not product_name:
        raise HTTPException(
            status_code=404,
            detail=f"Продукт '{req.product_id}' не найден. "
            f"Доступные: {list(ID_TO_PRODUCT.keys())}",
        )

    try:
        result = run_pipeline(product_name, str(req.date_from), str(req.date_to))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка пайплайна: {e}")

    return result


app.mount(
    "/",
    StaticFiles(directory=str(ROOT / "frontend"), html=True),
    name="frontend",
)
