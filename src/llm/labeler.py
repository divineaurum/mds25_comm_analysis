"""
Разметка кластера: промпт-сборщик + вызов LLM + парсинг JSON-ответа
"""

import json
import random
import re
from .llm_client import call_ollama

SYS_PROMPT = """Ты — опытный аналитик клиентского опыта в телекоммуникационной компании МТС.
Ты анализируешь открытые текстовые ответы из опроса NPS — клиенты-критики пишут, что им не понравилось.
Твоя задача — дать кластерам осмысленные названия, понятные продуктовым менеджерам.

Правила:
- Название темы должно быть конкретным (3-7 слов), без воды и лишнего текста.
- Severity: "высокая" — если клиент готов уйти или уже ушёл; "средняя" — ощутимая проблема; "низкая" — пожелание.
- Если тема не вписывается ни в одну из существующих категорий — это ценная находка, укажи "новая тема".
- Отвечай ТОЛЬКО JSON-объектом без markdown-блоков и лишнего текста."""


def build_cluster_info(
    texts: list[str],
    topic_id: int,
    keywords: list[str],
    product: str,
    n_comments: int | None = None,
):
    """
    Собирает информацию о кластере для передачи в LLM
    """
    size = len(texts)
    if n_comments is None:
        n_comments = min(15, max(5, size // 15))
    random.seed(42)
    sample = random.sample(texts, min(n_comments, size))
    return {
        "topic_id": topic_id,
        "keywords": keywords,
        "comments": sample,
        "size": size,
        "product": product,
    }


def build_prompt(info: dict, categories_str: str, rag_context: str):
    keywords_str = ", ".join(info["keywords"][:8])
    comments_str = "\n".join(f"- {c.strip()}" for c in info["comments"])
    return f"""Контекст продукта:
{rag_context}

Ниже — группа похожих комментариев клиентов, объединённых автоматической кластеризацией.
Ключевые слова кластера: {keywords_str}

Комментарии:
{comments_str}

Задача:
1. Дай короткое название этой теме (3-7 слов), понятное продуктовому менеджеру.
2. Опиши суть проблемы в 1-2 предложениях.
3. Сопоставь тему с одной из существующих категорий ТОЛЬКО при прямом тематическом совпадении
   (та же причина и явление, не просто смежные темы): {categories_str}.
   Если точного совпадения нет — обязательно укажи «новая тема» и объясни в rationale,
   чем она отличается от ближайшей существующей категории.
4. Оцени критичность: высокая / средняя / низкая.

Ответь строго в формате JSON:
{{"topic_name": "...", "description": "...", "existing_category": "... или новая тема", "is_new_topic": true/false, "rationale": "...", "severity": "..."}}"""


def parse_json_response(raw: str) -> dict | None:
    match = re.search(r"```json\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(candidate + "}")
            except Exception:
                pass
    return None


def label_cluster(
    info: dict,
    categories_str: str,
    rag_context: str,
) -> dict:
    """
    Вызывает LLM для разметки кластера

    Возвращает dict:
        parsed - распарсенный JSON (или None при ошибке парсинга)
        raw - сырой текст ответа модели
        elapsed - время генерации (сек)
        tokens - кол-во токенов
        valid_json - True, если парсинг успешен
    """
    prompt = build_prompt(info, categories_str, rag_context)
    result = call_ollama(prompt, system_prompt=SYS_PROMPT)
    parsed = parse_json_response(result["content"])

    if parsed:
        cat = (parsed.get("existing_category") or "").lower()
        parsed["is_new_topic"] = "новая тема" in cat or "new_topic" in cat

    return {
        "parsed": parsed,
        "raw": result["content"],
        "elapsed": result["elapsed"],
        "tokens": result["tokens"],
        "valid_json": parsed is not None,
    }
