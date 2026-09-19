#!/usr/bin/env python
"""End-to-end run: load data -> embed -> UMAP + KMeans -> save figure and metrics.

Example:
    python scripts/run_pipeline.py --n-samples 3000 --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from newssearch import cluster_embeddings, evaluate_clustering
from newssearch.config import BATCH_SIZE, MODEL_NAME, N_CLUSTERS, N_SAMPLES, RANDOM_STATE
from newssearch.data import load_ag_news
from newssearch.embeddings import embed_texts, load_model
from newssearch.viz import plot_cluster_comparison, project_2d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    p.add_argument("--model", default=MODEL_NAME, help="any sentence-transformers model name")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--clean-text", action="store_true", help="strip HTML/entity artifacts first")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    df = load_ag_news(args.n_samples, seed=args.seed, clean=args.clean_text)
    print(f"Loaded {len(df)} articles:\n{df['label_name'].value_counts()}\n")

    cache_name = f"emb_{args.model.replace('/', '_')}_{args.n_samples}_{args.seed}"
    cache_name += "_clean.npy" if args.clean_text else ".npy"
    embeddings = embed_texts(
        df["text"].tolist(),
        model=load_model(args.model),
        batch_size=args.batch_size,
        cache_path=args.cache_dir / cache_name,
    )
    print("Embeddings shape:", embeddings.shape)

    clusters = cluster_embeddings(embeddings, n_clusters=args.n_clusters, seed=args.seed)
    metrics = evaluate_clustering(df["label"], clusters.labels)
    print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
    print(f"Normalized Mutual Info: {metrics['normalized_mutual_info']:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_2d = project_2d(embeddings, seed=args.seed)
    plot_cluster_comparison(
        embeddings_2d,
        df["label"].to_numpy(),
        clusters.labels,
        metrics["adjusted_rand_index"],
        save_path=args.output_dir / "cluster_comparison.png",
    )

    run_info = {**vars(args), "output_dir": str(args.output_dir), "cache_dir": str(args.cache_dir)}
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"config": run_info, "metrics": metrics}, indent=2)
    )
    print(f"Saved figure and metrics to {args.output_dir}/")


if __name__ == "__main__":
    main()
