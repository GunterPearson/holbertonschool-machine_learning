#!/usr/bin/env python3
"""word embedding"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix"""
    cont_vector = CountVectorizer(vocabulary=vocab)
    embed = cont_vector.fit_transform(sentences)
    features = cont_vector.get_feature_names()
    return embed.toarray(), features
