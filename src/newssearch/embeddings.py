"""Sentence-embedding helpers, with optional on-disk caching."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from newssearch.config import BATCH_SIZE, MODEL_NAME


class Encoder(Protocol):
    """Anything with a sentence-transformers-style ``encode`` method."""

    def encode(self, sentences: Sequence[str], **kwargs) -> np.ndarray: ...


def load_model(model_name: str = MODEL_NAME) -> Encoder:
    """Load a sentence-transformers model (downloaded on first use)."""
    from sentence_transformers import SentenceTransformer  # lazy: heavy import

    return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str],
    model: Encoder | None = None,
    batch_size: int = BATCH_SIZE,
    show_progress_bar: bool = True,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """Encode ``texts`` into an ``(n, d)`` float array.

    If ``cache_path`` is given and exists, embeddings are loaded from it instead of
    recomputed; otherwise they are computed and saved there. The cache is validated
    only by row count, so delete it if you change the texts or the model.
    """
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        cached = np.load(cache)
        if cached.shape[0] == len(texts):
            return cached

    model = model or load_model()
    embeddings = model.encode(
        list(texts), show_progress_bar=show_progress_bar, batch_size=batch_size
    )
    embeddings = np.asarray(embeddings)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, embeddings)
    return embeddings
