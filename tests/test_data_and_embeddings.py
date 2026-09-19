import numpy as np

from newssearch.data import clean_text
from newssearch.embeddings import embed_texts


def test_clean_text_fixes_ag_news_artifacts():
    assert clean_text("Will Putin #39;s Power Play Make Russia Safer?") == (
        "Will Putin's Power Play Make Russia Safer?"
    )
    assert clean_text("India extended a  quot;hand of friendship quot; to Pakistan") == (
        'India extended a "hand of friendship" to Pakistan'
    )
    assert clean_text("Wall St. up &lt;b&gt;...&lt;/b&gt; on earnings") == (
        "Wall St. up ... on earnings"
    )
    assert clean_text("sold for \\$20 million") == "sold for $20 million"


def test_clean_text_leaves_plain_text_alone():
    assert clean_text("Stocks slip on GM earnings") == "Stocks slip on GM earnings"


def test_embed_texts_uses_and_validates_cache(encoder, corpus, tmp_path):
    cache = tmp_path / "emb.npy"
    first = embed_texts(corpus, model=encoder, show_progress_bar=False, cache_path=cache)
    assert cache.exists()

    class Exploding:
        def encode(self, *a, **k):
            raise AssertionError("should have used the cache")

    second = embed_texts(corpus, model=Exploding(), cache_path=cache)
    assert np.array_equal(first, second)

    # A cache with the wrong number of rows must be recomputed, not trusted.
    third = embed_texts(corpus[:2], model=encoder, show_progress_bar=False, cache_path=cache)
    assert third.shape[0] == 2
