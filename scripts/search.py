#!/usr/bin/env python
"""Semantic search over a sample of AG News from the command line.

Examples:
    python scripts/search.py "an athlete winning a gold medal"
    python scripts/search.py            # interactive prompt; empty line to quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from newssearch import SemanticSearchIndex
from newssearch.config import MODEL_NAME, N_SAMPLES, RANDOM_STATE
from newssearch.data import load_ag_news
from newssearch.embeddings import embed_texts, load_model
from newssearch.search import format_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("query", nargs="?", help="search query; omit for interactive mode")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--clean-text", action="store_true", help="strip HTML/entity artifacts first")
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_ag_news(args.n_samples, seed=args.seed, clean=args.clean_text)
    model = load_model(args.model)

    cache_name = f"emb_{args.model.replace('/', '_')}_{args.n_samples}_{args.seed}"
    cache_name += "_clean.npy" if args.clean_text else ".npy"
    embeddings = embed_texts(
        df["text"].tolist(),
        model=model,
        cache_path=args.cache_dir / cache_name,
        show_progress_bar=False,
    )
    index = SemanticSearchIndex(df["text"], embeddings, model, categories=df["label_name"])

    if args.query:
        print(format_results(args.query, index.search(args.query, args.top_k)))
        return

    print(f"Indexed {len(index)} articles. Enter a query (empty line to quit).")
    while query := input("\n> ").strip():
        print(format_results(query, index.search(query, args.top_k)))


if __name__ == "__main__":
    main()
