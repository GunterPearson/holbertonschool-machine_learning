#!/usr/bin/env python3
"""Q/A ChatBot module"""
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from transformers import BertTokenizer


module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
semantic_model = hub.load(module_url)
tz = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)
model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')


def question_answer(coprus_path):
    """answers questions from multiple reference texts"""
    while True:
        val = input("Q: ")
        exit_list = ['exit', 'quit', 'goodbye', 'bye']
        if val.lower() in exit_list:
            print("A: Goodbye")
            break
        reference = semantic_search(coprus_path, val)
        answer = q_answer(val, reference)
        if answer is None or answer is "":
            answer = "Sorry, I do not understand your question."
        print("A: {}".format(answer))


def semantic_search(corpus_path, sentence):
    """preforms semantice search"""
    documents = [sentence]
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md") is False:
            continue
        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
            documents.append(f.read())
    embed = semantic_model(documents)
    corr = np.inner(embed, embed)
    # We use 0 because that is the correlation line
    # that matches our sentence if you were looking
    # at graph with the first index being itself so we must remove it
    close = np.argmax(corr[0, 1:])
    similarity = documents[close + 1]
    return similarity


def q_answer(question, reference):
    """finds a snippet of text within a document to answer question"""
    question_ts = tz.tokenize(question)
    paragraph_ts = tz.tokenize(reference)
    tokens = ['[CLS]'] + question_ts + ['[SEP]'] + paragraph_ts + ['[SEP]']
    word_ids = tz.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(word_ids)
    type_ids = [0] * (1 + len(question_ts) + 1) + [1] * (len(paragraph_ts) + 1)

    word_ids, input_mask, type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (word_ids, input_mask, type_ids))
    outputs = model([word_ids, input_mask, type_ids])
    # using [1:] will enforce an answer. outputs[0][0][0] isignored
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tz.convert_tokens_to_string(answer_tokens)
    return answer
