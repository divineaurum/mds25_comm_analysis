"""
Поиск RAG-контекста по продукту
"""

from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path(__file__).parent.parent.parent
CHROMA_DIR = ROOT / "data" / "chroma_db_full"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "rag_product_context"

_collection = None


def get_collection(
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
):
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        ef = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        try:
            _collection = client.get_collection(
                name=collection_name,
                embedding_function=ef,
            )
        except Exception:
            raise RuntimeError(
                f"Коллекция '{collection_name}' не найдена в {chroma_dir}"
            )
    return _collection


def retrieve_context(
    collection,
    product_id: str,
    query: str,
    top_k: int = 3,
) -> dict:
    """
    Достаёт описание продукта (по точному ID) и top_k ближайших категорий жалоб

    Возвращает dict:
        product_description - текст описания продукта
        categories - список {name, description, similarity}
        rag_context - готовая строка для подстановки в промпт
        categories_str - перечисление категорий через '; '
    """
    prod_result = collection.get(
        ids=[f"{product_id}_description"],
        include=["documents", "metadatas"],
    )
    product_description = (
        prod_result["documents"][0] if prod_result["documents"] else ""
    )

    categories = []
    try:
        query_result = collection.query(
            query_texts=[query],
            where={
                "$and": [
                    {"product_id": {"$eq": product_id}},
                    {"doc_type": {"$eq": "category"}},
                ]
            },
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if query_result["documents"] and query_result["documents"][0]:
            for doc, meta, dist in zip(
                query_result["documents"][0],
                query_result["metadatas"][0],
                query_result["distances"][0],
            ):
                categories.append(
                    {
                        "name": meta.get("name", ""),
                        "description": doc,
                        "similarity": round(1 - dist, 4),
                    }
                )
    except Exception as e:
        print(f"RAG-запрос для '{product_id}' не удался: {e}")

    categories_str = "; ".join(c["name"] for c in categories)
    context_parts = [product_description] if product_description else []
    if categories_str:
        context_parts.append(f"Известные категории жалоб: {categories_str}")

    return {
        "product_description": product_description,
        "categories": categories,
        "rag_context": "\n\n".join(context_parts),
        "categories_str": categories_str,
    }
