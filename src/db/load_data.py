"""
Создание таблицы comments и загрузка обезличенных комментариев из CSV в SQLite
"""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CSV_PATH = ROOT / "data" / "processed" / "comments_anonymized_full.csv"
DB_PATH = ROOT / "data" / "processed" / "comments_full.db"


def load_data(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH):
    df = pd.read_csv(csv_path, encoding="utf-8", sep=";")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            date     DATE,
            product  TEXT,
            platform TEXT,
            text     TEXT
        )
    """
    )

    cursor.execute("DELETE FROM comments")

    records = [
        (
            str(row["date"])[:10] if pd.notna(row["date"]) else None,
            row["product"],
            row["platform"] if pd.notna(row.get("platform")) else None,
            row["text"],
        )
        for _, row in df.iterrows()
    ]

    cursor.executemany(
        "INSERT INTO comments (date, product, platform, text) VALUES (?, ?, ?, ?)",
        records,
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    load_data()
