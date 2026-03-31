"""
Инициализация ChromaDB из файлов в rag_docs/: читает description.txt и categories.json для каждого продукта.

Запуск:
    python src/llm/populate_rag.py  # создать / добавить (upsert)
    python src/llm/populate_rag.py --recreate  # удалить коллекцию и создать заново
"""

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path(__file__).parent.parent.parent
RAG_DOCS_DIR = ROOT / "rag_docs"
CHROMA_DIR = ROOT / "data" / "chroma_db_full"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "rag_product_context"


def populate(recreate: bool = False) -> int:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    if recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Коллекция '{COLLECTION_NAME}' удалена.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    product_dirs = sorted(d for d in RAG_DOCS_DIR.iterdir() if d.is_dir())
    ids, documents, metadatas = [], [], []

    for product_dir in product_dirs:
        product_id = product_dir.name
        print(f"[{product_id}]")

        desc_path = product_dir / "description.txt"
        if desc_path.exists():
            text = desc_path.read_text(encoding="utf-8").strip()
            ids.append(f"{product_id}_description")
            documents.append(text)
            metadatas.append({"product_id": product_id, "doc_type": "description"})
            print(f"description.txt - загружено ({len(text)} символов)")

        cat_path = product_dir / "categories.json"
        if cat_path.exists():
            categories = json.loads(cat_path.read_text(encoding="utf-8"))
            for cat in categories:
                ids.append(cat["id"])
                documents.append(f"{cat['name']}: {cat['description']}")
                metadatas.append(
                    {
                        "product_id": product_id,
                        "doc_type": "category",
                        "name": cat["name"],
                    }
                )
            print(f"categories.json - загружено {len(categories)} категорий")

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    total = len(ids)
    print(f"Итого загружено документов: {total}")
    print(f"ChromaDB: {CHROMA_DIR}")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инициализация ChromaDB для RAG")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Удалить коллекцию и создать заново (полный пересчёт эмбеддингов)",
    )
    args = parser.parse_args()
    populate(recreate=args.recreate)
