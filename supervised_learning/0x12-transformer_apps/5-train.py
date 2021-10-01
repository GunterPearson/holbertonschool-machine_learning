#!/usr/bin/env python3
"""Transformer Application"""
import tensorflow as tf
import tensorflow_datasets as tfds
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """class CustomSchedule"""
    def __init__(self, d_model, warmup_steps=4000):
        """constructor"""
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """call method"""
        call1 = tf.math.rsqrt(step)
        call2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(call1, call2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """creates and trains a transformer model for machine translation"""
    data = Dataset(batch_size, max_len)
    input_size = data.tokenizer_pt.vocab_size + 2
    target_size = data.tokenizer_en.vocab_size + 2
    transformer = Transformer(N, dm, h, hidden,
                              input_size, target_size,
                              max_len, max_len)

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    losses = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                           reduction='none')

    def loss_function(actual, prediction):
        """Calculate loss"""
        mask = tf.math.logical_not(tf.math.equal(actual, 0))
        loss = losses(actual, prediction)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    for epoch in range(epochs):
        batch = 0
        for (input, target) in data.data_train:
            target_input = target[:, :-1]
            target_actual = target[:, 1:]
            encoder_mask, look_ahead_mask, decoder_mask = create_masks(
                input, target_input)
            with tf.GradientTape() as tape:
                prediction = transformer(input, target_input, True,
                                         encoder_mask, look_ahead_mask,
                                         decoder_mask)
                loss = loss_function(target_actual, prediction)
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(
                gradients, transformer.trainable_variables))
            t_loss = train_loss(loss)
            t_accuracy = train_accuracy(target_actual, prediction)
            if batch % 50 is 0:
                print("Epoch {}, batch {}: loss {} accuracy {}".format(
                    epoch + 1, batch, t_loss, t_accuracy))
            batch += 1
        print("Epoch {}: loss {} accuracy {}".format(
            epoch, t_loss, t_accuracy))
    return transformer
