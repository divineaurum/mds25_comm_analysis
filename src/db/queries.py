"""
Запросы к SQLite: список продуктов и выборка комментариев по продукту и периоду
"""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "processed" / "comments_full.db"


def get_products(db_path: Path = DB_PATH) -> list[str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT product FROM comments ORDER BY product")
    products = [row[0] for row in cursor.fetchall()]
    conn.close()
    return products


def get_product_stats(product: str, db_path: Path = DB_PATH) -> dict:
    """
    Возвращает количество комментариев и диапазон дат для продукта
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*), MIN(date), MAX(date)
        FROM comments
        WHERE product = ? AND text IS NOT NULL AND trim(text) != ''
        """,
        (product,),
    )
    count, date_min, date_max = cursor.fetchone()
    conn.close()
    return {"total_comments": count or 0, "date_min": date_min, "date_max": date_max}


def get_comments(
    product: str,
    date_from: str,
    date_to: str,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """
    Возвращает DataFrame комментариев для продукта за период [date_from, date_to]
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT id, date, product, platform, text
        FROM comments
        WHERE product = ?
          AND date >= ?
          AND date <= ?
          AND text IS NOT NULL
          AND trim(text) != ''
        ORDER BY date
        """,
        conn,
        params=(product, date_from, date_to),
    )
    conn.close()
    return df
