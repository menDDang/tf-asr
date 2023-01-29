from __future__ import absolute_import

from typing import Union

import tensorflow as tf

class TimeReduction(tf.keras.layers.Layer):
    def __init__(
        self,
        factor: int,
        name: str = "TimeReduction",
        **kwargs
    ):
        super(TimeReduction, self).__init__(name=name, **kwargs)

        self.time_reduction_factor = factor

    def call(
        self,
        inputs,
        **kwargs
    ):
        B, T, D = inputs.shape
        paddings = tf.cast(
            tf.math.ceil(
                (T / self.time_reduction_factor) * self.time_reduction_factor
            ),
            dtype=tf.int32
        ) - T

        outputs = tf.pad(
            inputs,
            [[0, 0], [0, paddings], [0, 0]]
        )
        outputs = tf.reshape(
            outputs,
            [B, -1, D * self.time_reduction_factor]
        )


class Conv2dSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        strides: Union[list, tuple, int] = 2,
        kernel_size: Union[int, list, tuple] = 3,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="Conv2dSubsampling",
        **kwargs
    ):
        super(Conv2dSubsampling, self).__init__(name=name, **kwargs)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            activation="relu",
            name=f"{name}_conv1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            activation="relu",
            name=f"{name}_conv2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.time_reduction_factor = self.conv1.strides[0] * self.conv2.strides[0]

    def call(
        self,
        inputs,
        training=False,
        **kwargs
    ):
        outputs = self.conv1(inputs, training=training)
        outputs = self.conv2(outputs, training=training)

        B, T, D1, D2 = outputs.shape
        outputs = tf.reshape(outputs, [B, -1, D1 * D2])
        return outputs