"""Test fixtures. A tiny deterministic bag-of-words encoder stands in for the real model,
so the test suite needs no network access and no heavy ML dependencies."""

import zlib

import numpy as np
import pytest

DIM = 64


class StubEncoder:
    """Hashes each lowercase word into a fixed-size count vector."""

    def encode(self, sentences, **kwargs):
        out = np.zeros((len(sentences), DIM))
        for row, sentence in enumerate(sentences):
            for word in sentence.lower().split():
                out[row, zlib.crc32(word.encode()) % DIM] += 1.0
        return out


@pytest.fixture
def encoder():
    return StubEncoder()


@pytest.fixture
def corpus():
    return [
        "stocks fall as earnings disappoint investors",
        "olympic athlete wins gold medal in track",
        "new software release from technology company",
        "election results announced in foreign country",
    ]
