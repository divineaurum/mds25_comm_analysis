"""
Обёртка для Ollama API
"""

import time
import requests

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"
GENERATION_CONFIG = {
    "temperature": 0.0,
    "seed": 42,
    "top_p": 0.9,
    "num_predict": 300,
}


def call_ollama(
    user_message: str,
    system_prompt: str | None = None,
    model: str = MODEL_NAME,
    config: dict = GENERATION_CONFIG,
) -> dict:
    """
    Отправляет запрос в Ollama и возвращает словарь:
        content - текст ответа модели
        elapsed - время генерации (в секундах)
        tokens - кол-во токенов
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    t0 = time.time()
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False, "options": config},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()

    return {
        "content": data["message"]["content"],
        "elapsed": round(time.time() - t0, 2),
        "tokens": data.get("eval_count", 0),
    }
