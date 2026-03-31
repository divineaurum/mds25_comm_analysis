"""
Кластеризация текстов через BERTopic с multilingual-e5-large эмбеддингами
"""

import numpy as np
import pandas as pd
import torch
import nltk
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
N_COMPONENTS_UMAP = 5


def _auto_min_cluster_size(n: int) -> int:
    return max(7, n // 60)


def _auto_min_samples(min_cluster_size: int) -> int:
    return max(3, min_cluster_size // 3)


_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _embedding_model


def cluster(
    texts: list[str],
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Кластеризует тексты с помощью BERTopic

    Возвращает:
        df - DataFrame с колонками 'text' и 'topic' (-1 - шум)
        keywords_dict - {str(topic_id): [слово1, слово2, ...]}
    """
    if min_cluster_size is None:
        min_cluster_size = _auto_min_cluster_size(len(texts))
    if min_samples is None:
        min_samples = _auto_min_samples(min_cluster_size)
    print(
        f"min_cluster_size={min_cluster_size}, min_samples={min_samples} (n={len(texts)})"
    )

    model = _get_embedding_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    texts_prefixed = ["passage: " + t for t in texts]
    embeddings = model.encode(
        texts_prefixed,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device,
    )

    try:
        russian_sw = nltk.corpus.stopwords.words("russian")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        russian_sw = nltk.corpus.stopwords.words("russian")

    umap_model = UMAP(
        n_components=N_COMPONENTS_UMAP,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words=russian_sw,
        min_df=2,
        ngram_range=(1, 2),
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        language="multilingual",
        calculate_probabilities=True,
        verbose=True,
    )

    topic_assignments, _ = topic_model.fit_transform(texts, embeddings)

    df = pd.DataFrame({"text": texts, "topic": topic_assignments})

    topic_info = topic_model.get_topic_info()
    keywords_dict: dict[str, list[str]] = {}
    for _, row in topic_info[topic_info["Topic"] != -1].iterrows():
        tid = int(row["Topic"])
        keywords_dict[str(tid)] = [w for w, _ in topic_model.get_topic(tid)]

    n_clusters = len(keywords_dict)
    n_noise = int((np.array(topic_assignments) == -1).sum())
    print(f"Кластеров: {n_clusters} | Шум: {n_noise} / {len(texts)}")

    return df, keywords_dict
