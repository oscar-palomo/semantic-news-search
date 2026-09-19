import numpy as np

from newssearch import cluster_embeddings, evaluate_clustering


def make_blobs(n_per=50, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0], [10.0, -10.0]])
    X = np.vstack([c + rng.normal(scale=0.5, size=(n_per, 2)) for c in centers])
    y = np.repeat(np.arange(4), n_per)
    return X, y


def test_kmeans_recovers_well_separated_clusters():
    X, y = make_blobs()
    result = cluster_embeddings(X, n_clusters=4, seed=42)
    assert result.labels.shape == (len(X),)
    assert evaluate_clustering(y, result.labels)["adjusted_rand_index"] > 0.99


def test_clustering_is_deterministic_for_a_seed():
    X, _ = make_blobs()
    a = cluster_embeddings(X, seed=7).labels
    b = cluster_embeddings(X, seed=7).labels
    assert np.array_equal(a, b)


def test_metrics_ignore_cluster_numbering():
    y = np.array([0, 0, 1, 1, 2, 2])
    permuted = np.array([2, 2, 0, 0, 1, 1])
    m = evaluate_clustering(y, permuted)
    assert m["adjusted_rand_index"] == 1.0
    assert m["normalized_mutual_info"] == 1.0
