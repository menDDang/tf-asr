from __future__ import absolute_import

import tensorflow as tf


class PreEmphasis(tf.keras.layers.Layer):
    
    def __init__(self, preemph: float=0.97, **kwargs):
        assert (0 < preemph < 1)
        super().__init__(**kwargs)
        self.preemph = preemph

    def call(self, x: tf.Tensor) -> tf.Tensor:
        time_length = tf.gather(tf.shape(x), indices=1)
        s0 = tf.slice(x, begin=[0, 0], size=[-1, 1])
        s1 = tf.slice(x, begin=[0, 1], size=[-1, time_length-1])
        s1 -= self.preemph * tf.slice(x, begin=[0, 0], size=[-1, time_length-1])
        return tf.concat([s0, s1], axis=-1)