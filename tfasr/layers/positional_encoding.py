from __future__ import absolute_import
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        alpha: int=1,
        beta: int=0,
        name="positional_encoding",
        **kwargs
    ):
        super(PositionalEncoding, self).__init__(
            name=name,
            **kwargs
        )
        self.alpha = alpha
        self.beta = beta
        
    def build(
        self,
        input_shape
    ):
        d_model = input_shape[-1]
        assert d_model % 2 == 0, f"Input last dim must be even: {d_model}"

    def call(
        self,
        inputs,
        **kwargs
    ):
        # input shape : [B, T, D]
        max_len = tf.shape(inputs)[1]
        d_model = inputs.shape[2]
        
        max_len = max_len * self.alpha + self.beta

        pos = tf.expand_dims(
            tf.range(max_len - 1, -1, -1.0, dtype=tf.float32),
            axis=-1
        )
        index = tf.expand_dims(
            tf.range(0, d_model, dtype=tf.float32),
            axis=0
        )

        positional_encoding = \
            pos / tf.pow(10000.0, (2 * (index // 2)) / d_model)

        # sin, cos will be [max_len, size // 2]
        # we add 0 betweetn numbers by using padding and reshape
        sin = tf.pad(
            tf.expand_dims(tf.sin(positional_encoding[:, 0::2]), axis=-1),
            [[0, 0], [0, 0], [0, 1]],
            mode="CONSTANT",
            constant_values=0
        )
        sin = tf.reshape(sin, [max_len, d_model])
        cos = tf.pad(
            tf.expand_dims(tf.cos(positional_encoding[:, 1::2]), axis=-1),
            [[0, 0], [0, 0], [1, 0]],
            mode="CONSTANT",
            constant_values=0
        )
        cos = tf.reshape(cos, [max_len, d_model])

        # Then add sin and cos, which results in [time, size]
        positional_encoding = tf.add(sin, cos)

        # [1, time, size]
        positional_encoding = tf.expand_dims(positional_encoding, axis=0)

        return tf.cast(positional_encoding, dtype=inputs.dtype)