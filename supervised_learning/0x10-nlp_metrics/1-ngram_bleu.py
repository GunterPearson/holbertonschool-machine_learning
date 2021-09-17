#!/usr/bin/env python3
"""Natural Language Processing - Evaluation Metrics"""
import numpy as np


def n_gram(sentence, n):
    """Tokenize sentence into grams"""
    if n <= 1:
        return sentence
    step = n - 1

    result = sentence[:-step]
    for i in range(len(result)):
        for j in range(step):
            result[i] += ' ' + sentence[i + 1 + j]
    return result


def ngram_bleu(references, sentence, n):
    """calculates the n-gram BLEU score for a sentence"""
    c = len(sentence)
    rs = [len(r) for r in references]

    sentence = n_gram(sentence, n)
    references = list(map(lambda ref: n_gram(ref, n), references))
    flat = set([gram for ref in references for gram in ref])

    top = 0
    for gram in flat:
        if gram in sentence:
            top += 1
    precision = top / len(sentence)

    best_match = None
    for i, ref in enumerate(references):
        if best_match is None:
            best_match = ref
            r_idx = i
        best_diff = abs(len(best_match) - len(sentence))
        if abs(len(ref) - len(sentence)) < best_diff:
            best_match = ref
            r_idx = i

    r = rs[r_idx]
    if c > r:
        brevity_penality = 1
    else:
        brevity_penality = np.exp(1 - r / c)

    bleu_score = brevity_penality * precision
    return bleu_score
