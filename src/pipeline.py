"""
Полный пайплайн: load -> cluster -> RAG+LLM -> JSON

Запуск из корня репозитория:
    python src/pipeline.py --product "Аудио-платформа" --date-from 2025-01-01 --date-to 2025-12-31

Результаты сохраняются в results/pipeline_{product_slug}_{date_from}_{date_to}.json
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PRODUCT_TO_ID = {
    "Видео-платформа": "video_platform",
    "Аудио-платформа": "audio_platform",
    "Маркетплейс": "marketplace",
    "Финансовый продукт": "financial_product",
    "Личный кабинет": "personal_account",
}


def run_pipeline(product: str, date_from: str, date_to: str) -> dict:
    from src.db.queries import get_comments
    from src.clustering.bertopic_pipeline import cluster
    from src.llm.rag_retriever import get_collection, retrieve_context
    from src.llm.labeler import build_cluster_info, label_cluster
    from src.llm.llm_client import get_vram_mb

    product_id = PRODUCT_TO_ID.get(product)
    if product_id is None:
        raise ValueError(
            f"Неизвестный продукт: '{product}'.\n"
            f"Доступные: {list(PRODUCT_TO_ID.keys())}"
        )

    print(f"\n{'='*60}")
    print(f"Продукт : {product}")
    print(f"Период  : {date_from} — {date_to}")
    print(f"{'='*60}")

    # 1. Загрузка комментариев из БД
    df = get_comments(product, date_from, date_to)
    if df.empty:
        print("Нет комментариев за выбранный период.")
        return {
            "product": product,
            "period": f"{date_from} — {date_to}",
            "total_comments": 0,
            "topics": [],
        }
    print(f"Загружено комментариев: {len(df)}")

    # 2. Кластеризация BERTopic
    texts = df["text"].tolist()
    clustered_df, keywords_dict = cluster(texts)

    unique_topics = sorted(t for t in clustered_df["topic"].unique() if t != -1)
    if not unique_topics:
        print("[!] BERTopic не нашёл ни одного кластера (все в шуме).")
        print("    Попробуй уменьшить min_cluster_size в bertopic_pipeline.py")
        return {
            "product": product,
            "period": f"{date_from} — {date_to}",
            "total_comments": len(df),
            "topics": [],
        }
    print(f"Кластеров для разметки: {len(unique_topics)}")

    # 3. Инициализация RAG-коллекции (один раз)
    collection = get_collection()
    vram_mb = get_vram_mb()

    # 4. Разметка каждого кластера через LLM + RAG
    total_elapsed = 0.0
    total_tokens_out = 0
    total_tokens_in = 0
    skipped = 0
    topics = []
    for topic_id in unique_topics:
        cluster_texts = clustered_df[clustered_df["topic"] == topic_id]["text"].tolist()
        keywords = keywords_dict.get(str(topic_id), [])

        rag = retrieve_context(collection, product_id, query=" ".join(keywords[:5]))
        info = build_cluster_info(cluster_texts, topic_id, keywords, product)
        result = label_cluster(info, rag["categories_str"], rag["rag_context"])

        total_elapsed += result["elapsed"]
        total_tokens_out += result["tokens"]
        total_tokens_in += result["prompt_tokens"]

        if not result["valid_json"]:
            print(f"Кластер {topic_id}: невалидный JSON - пропущен")
            print(f"Полученный ответ: {result['raw'][:120]}...")
            skipped += 1
            continue

        parsed = result["parsed"]
        topics.append(
            {
                "topic_name": parsed.get("topic_name", ""),
                "description": parsed.get("description", ""),
                "existing_category": parsed.get("existing_category", ""),
                "is_new_topic": parsed.get("is_new_topic", False),
                "rationale": parsed.get("rationale", ""),
                "severity": parsed.get("severity", ""),
                "count": len(cluster_texts),
                "percentage": round(len(cluster_texts) / len(df) * 100, 1),
                "example_comments": info["comments"][:3],
            }
        )

        flag = "[NEW]" if parsed.get("is_new_topic") else "     "
        print(
            f"  {flag} [{topic_id}] {parsed.get('topic_name', '?')}"
            f" — {len(cluster_texts)} комм. ({result['elapsed']}с)"
        )

    # 5. Сборка и сохранение результата
    output = {
        "product": product,
        "period": f"{date_from} — {date_to}",
        "total_comments": len(df),
        "topics": sorted(topics, key=lambda t: t["count"], reverse=True),
        "meta": {
            "vram_mb": vram_mb,
            "total_elapsed_sec": round(total_elapsed, 2),
            "total_tokens_out": total_tokens_out,
            "total_tokens_in": total_tokens_in,
            "clusters_labeled": len(topics),
            "clusters_skipped": skipped,
        },
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    slug = re.sub(r"[^\w]", "_", product.lower())
    out_path = results_dir / f"pipeline_{slug}_{date_from}_{date_to}.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Сохранено: {out_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Пайплайн анализа комментариев")
    parser.add_argument("--product", type=str, help="Название продукта")
    parser.add_argument("--date-from", type=str, default="2025-09-01", dest="date_from")
    parser.add_argument(
        "--date-to", type=str, default=str(date.today()), dest="date_to"
    )
    args = parser.parse_args()

    if args.product:
        run_pipeline(args.product, args.date_from, args.date_to)
    else:
        parser.print_help()

    sys.exit(0)
