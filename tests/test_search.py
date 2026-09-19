import pytest

from newssearch import SemanticSearchIndex
from newssearch.search import format_results


def build(encoder, corpus, categories=None):
    return SemanticSearchIndex(corpus, encoder.encode(corpus), encoder, categories=categories)


def test_top_result_is_most_similar(encoder, corpus):
    index = build(encoder, corpus)
    results = index.search("gold medal athlete", top_k=2)
    assert results[0].text == corpus[1]
    assert results[0].rank == 1
    assert results[0].score >= results[1].score


def test_top_k_is_respected_and_sorted(encoder, corpus):
    results = build(encoder, corpus).search("stocks earnings", top_k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_larger_than_corpus_returns_everything(encoder, corpus):
    assert len(build(encoder, corpus).search("stocks", top_k=100)) == len(corpus)


def test_categories_are_attached(encoder, corpus):
    cats = ["Business", "Sports", "Sci/Tech", "World"]
    top = build(encoder, corpus, categories=cats).search("gold medal", top_k=1)[0]
    assert top.category == "Sports"
    assert top.index == 1


def test_validation(encoder, corpus):
    with pytest.raises(ValueError):
        SemanticSearchIndex(corpus, encoder.encode(corpus[:2]), encoder)
    with pytest.raises(ValueError):
        SemanticSearchIndex(corpus, encoder.encode(corpus), encoder, categories=["x"])
    index = build(encoder, corpus)
    with pytest.raises(ValueError):
        index.search("   ")
    with pytest.raises(ValueError):
        index.search("stocks", top_k=0)


def test_format_results(encoder, corpus):
    index = build(encoder, corpus, categories=["Business", "Sports", "Sci/Tech", "World"])
    text = format_results("gold medal", index.search("gold medal", top_k=1))
    assert text.startswith('Query: "gold medal"')
    assert "(Sports)" in text
