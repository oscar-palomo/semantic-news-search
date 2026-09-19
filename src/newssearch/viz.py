"""2D projection and plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from newssearch.config import LABEL_NAMES, RANDOM_STATE


def project_2d(embeddings: np.ndarray, seed: int = RANDOM_STATE) -> np.ndarray:
    """UMAP projection to 2D. Used for plotting only, never for clustering."""
    from umap import UMAP  # lazy: heavy import

    return UMAP(n_components=2, random_state=seed).fit_transform(embeddings)


def plot_cluster_comparison(
    embeddings_2d: np.ndarray,
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    ari: float,
    label_names: dict[int, str] = LABEL_NAMES,
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    """Side-by-side scatter: true categories (left) vs KMeans clusters (right)."""
    true_labels = np.asarray(true_labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for label_val, label_name in label_names.items():
        mask = true_labels == label_val
        axes[0].scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1], s=8, alpha=0.6, label=label_name
        )
    axes[0].set_title("True news categories")
    axes[0].legend(markerscale=2)

    axes[1].scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], s=8, alpha=0.6, c=cluster_labels, cmap="tab10"
    )
    axes[1].set_title(f"KMeans clusters (unsupervised)\nARI vs true labels: {ari:.3f}")

    for ax in axes:
        ax.set_xlabel("UMAP dimension 1")
        ax.set_ylabel("UMAP dimension 2")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
    return fig
