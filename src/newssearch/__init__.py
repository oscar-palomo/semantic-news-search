"""Semantic search and unsupervised clustering over news text using sentence embeddings."""

from newssearch.clustering import ClusteringResult, cluster_embeddings, evaluate_clustering
from newssearch.config import LABEL_NAMES, RANDOM_STATE
from newssearch.search import SearchResult, SemanticSearchIndex

__all__ = [
    "LABEL_NAMES",
    "RANDOM_STATE",
    "ClusteringResult",
    "SearchResult",
    "SemanticSearchIndex",
    "cluster_embeddings",
    "evaluate_clustering",
]

__version__ = "0.1.0"
