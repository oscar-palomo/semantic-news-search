"""Loading and (optionally) cleaning the AG News dataset."""

from __future__ import annotations

import html
import re

import pandas as pd

from newssearch.config import DATASET_NAME, LABEL_NAMES, N_SAMPLES, RANDOM_STATE

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# AG News contains HTML entities whose leading "&" was replaced by a space, e.g.
# "Putin&#39;s" became "Putin #39;s" and "&quot;hand&quot;" became " quot;hand quot;".
# The optional leading whitespace in these patterns swallows that stray space.
_APOSTROPHE_RE = re.compile(r"\s?#39;")
_QUOTE_RE = re.compile(r"\s?quot;")


def clean_text(text: str) -> str:
    """Remove the HTML/entity artifacts found in AG News.

    Embedding models tolerate this noise reasonably well, so cleaning is *off* by
    default in the pipeline (to match the original notebook's numbers). It is useful
    for making search results readable. Only artifacts actually observed in the data
    are handled; anything else is left untouched rather than guessed at.
    """
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _APOSTROPHE_RE.sub("'", text)
    text = _QUOTE_RE.sub('"', text)
    text = text.replace("\\$", "$")
    return _WS_RE.sub(" ", text).strip()


def load_ag_news(
    n_samples: int = N_SAMPLES,
    seed: int = RANDOM_STATE,
    clean: bool = False,
    split: str = "train",
) -> pd.DataFrame:
    """Load a shuffled random sample of AG News as a DataFrame.

    Columns: ``text``, ``label`` (int), ``label_name`` (str).
    """
    from datasets import load_dataset  # imported lazily: heavy, and needs network

    dataset = load_dataset(DATASET_NAME)
    if n_samples > len(dataset[split]):
        raise ValueError(f"n_samples={n_samples} exceeds split size {len(dataset[split])}")

    df = dataset[split].shuffle(seed=seed).select(range(n_samples)).to_pandas()
    if clean:
        df["text"] = df["text"].map(clean_text)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    return df
