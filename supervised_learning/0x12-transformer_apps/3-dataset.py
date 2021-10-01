#!/usr/bin/env python3
"""Transformer Application"""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
    """loads and preps a dataset for machine translation"""
    def __init__(self, batch_size, max_len):
        """class Constructor"""
        data_train, data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train
        )
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

        def filter_max_length(x, y, max_len=max_len):
            """filter max len"""
            filtered = tf.logical_and(tf.size(x) <= max_len,
                                      tf.size(y) <= max_len)
            return filtered

        self.data_train = self.data_train.filter(filter_max_length)
        self.data_valid = self.data_valid.filter(filter_max_length)

        self.data_train = self.data_train.cache()
        data_size = sum(1 for _ in self.data_train)
        self.data_train = self.data_train.shuffle(data_size)

        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_valid = self.data_valid.padded_batch(batch_size)

        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset"""
        pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (p.numpy() for p, e in data), target_vocab_size=2**15
        )
        en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (e.numpy() for p, e in data), target_vocab_size=2**15
        )
        return pt, en

    def encode(self, pt, en):
        """encodes a translation into tokens"""
        pt_vs = self.tokenizer_pt.vocab_size
        en_vs = self.tokenizer_en.vocab_size

        pt_e = [pt_vs] + self.tokenizer_pt.encode(pt.numpy()) + [pt_vs + 1]
        en_e = [en_vs] + self.tokenizer_en.encode(en.numpy()) + [en_vs + 1]
        return pt_e, en_e

    def tf_encode(self, pt, en):
        """tensorflow wrapper for the encode instance method"""
        pt_encode, en_encode = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_encode.set_shape([None])
        en_encode.set_shape([None])
        return pt_encode, en_encode
