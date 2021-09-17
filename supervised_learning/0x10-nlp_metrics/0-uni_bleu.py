#!/usr/bin/env python3
"""Natural Language Processing - Evaluation Metrics"""
import numpy as np


def uni_bleu(references, sentence):
    """calculates the unigram BLEU score for a sentence"""
    s_len = len(sentence)
    r_len = []
    words = {}
    for word in references:
        r_len.append(len(word))
        for word in word:
            if word in sentence and word not in words.keys():
                words[word] = 1

    total = sum(words.values())
    index = np.argmin([abs(len(i) - s_len) for i in references])
    best_match = len(references[index])

    if s_len > best_match:
        brevity = 1
    else:
        brevity = np.exp(1 - float(best_match) / float(s_len))
    score = brevity * np.exp(np.log(total / s_len))

    return score
