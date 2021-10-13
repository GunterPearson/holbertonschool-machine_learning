#!/usr/bin/env python3
"""Q/A ChatBot module"""
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def semantic_search(corpus_path, sentence):
    """preforms semantice search"""
    documents = [sentence]
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md") is False:
            continue
        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
            documents.append(f.read())
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(module_url)
    embed = model(documents)
    corr = np.inner(embed, embed)
    # We use 0 because that is the correlation line
    # that matches our sentence if you were looking
    # at graph with the first index being itself so we must remove it
    close = np.argmax(corr[0, 1:])
    similarity = documents[close + 1]
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(corr[0], annot=True)
    # plt.show()
    return similarity
