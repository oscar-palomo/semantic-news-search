"""Unsupervised clustering of embeddings and comparison with ground-truth labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from newssearch.config import N_CLUSTERS, RANDOM_STATE


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    seed: int = RANDOM_STATE,
    n_init: int = 10,
) -> ClusteringResult:
    """Run KMeans on the full-dimensional embeddings (not on any 2D projection)."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    labels = kmeans.fit_predict(embeddings)
    return ClusteringResult(labels=labels, centers=kmeans.cluster_centers_, inertia=kmeans.inertia_)


def evaluate_clustering(true_labels, cluster_labels) -> dict[str, float]:
    """Agreement between clusters and known categories.

    Both metrics are invariant to how clusters are numbered. ARI is chance-adjusted
    (0 ~ random, 1 = identical partitions); NMI is in [0, 1].
    """
    return {
        "adjusted_rand_index": float(adjusted_rand_score(true_labels, cluster_labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(true_labels, cluster_labels)),
    }
