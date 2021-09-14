#!/usr/bin/env python3
"""word embedding"""


def gensim_to_keras(model):
    """Convert a genesim model to keras Embedding layer"""
    return model.wv.get_keras_embedding(train_embeddings=True)
