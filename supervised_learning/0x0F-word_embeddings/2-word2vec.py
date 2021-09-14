#!/usr/bin/env python3
"""word embedding"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """Creates and trains a Word2Vec model"""
    return Word2Vec(
        sentences,
        size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        iter=iterations,
        workers=workers,
        seed=seed,
        sg=not cbow
    )
