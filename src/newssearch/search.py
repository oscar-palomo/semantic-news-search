"""Cosine-similarity semantic search over a corpus of pre-computed embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from newssearch.embeddings import Encoder


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    text: str
    category: str | None = None
    index: int = -1


class SemanticSearchIndex:
    """Brute-force nearest-neighbour search over sentence embeddings.

    Brute force is exact and plenty fast for tens of thousands of documents. For
    millions, swap in an ANN library such as FAISS or hnswlib.
    """

    def __init__(
        self,
        texts: Sequence[str],
        embeddings: np.ndarray,
        model: Encoder,
        categories: Sequence[str] | None = None,
    ) -> None:
        if len(texts) != len(embeddings):
            raise ValueError(f"{len(texts)} texts but {len(embeddings)} embeddings")
        if categories is not None and len(categories) != len(texts):
            raise ValueError(f"{len(texts)} texts but {len(categories)} categories")
        self.texts = list(texts)
        self.embeddings = np.asarray(embeddings)
        self.model = model
        self.categories = list(categories) if categories is not None else None

    def __len__(self) -> int:
        return len(self.texts)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the ``top_k`` documents most similar to ``query``, best first."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        query_embedding = np.asarray(self.model.encode([query]))
        sims = cosine_similarity(query_embedding, self.embeddings)[0]
        top_idx = np.argsort(-sims)[:top_k]
        return [
            SearchResult(
                rank=rank,
                score=float(sims[i]),
                text=self.texts[i],
                category=self.categories[i] if self.categories is not None else None,
                index=int(i),
            )
            for rank, i in enumerate(top_idx, start=1)
        ]


def format_results(query: str, results: Sequence[SearchResult]) -> str:
    """Render results as plain text for the terminal or notebook."""
    lines = [f'Query: "{query}"', "-" * 60]
    for r in results:
        tag = f" ({r.category})" if r.category else ""
        lines.append(f"[{r.score:.3f}]{tag} {r.text}")
    return "\n".join(lines)
