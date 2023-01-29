from __future__ import absolute_import

import tensorflow as tf


class TimeDepthSeparableBlock(tf.keras.layers.Layer):
    """ Time Depth Separable Block 
    
    This class is based on paper:
        https://arxiv.org/pdf/1904.02619.pdf

    """
    def __init__(self, 
                 kernel_size: int, 
                 dropout_prob: float = 0.1, 
                 name="tds_block", 
                 **kwargs):
        super(TimeDepthSeparableBlock, self).__init__(**kwargs)

        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob
        
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()

        self.dropout2 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        # Parse input shape : [B, T, W, C]
        _, _, W, C = input_shape
        self.W = W
        self.C = C

        # Create sub block - 2D convolution over time
        self.sub_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=C,
                kernel_size=[self.kernel_size, 1],
                strides=[1, 1],
                padding='same'),
            tf.keras.layers.Activation('relu'),
        ])

        # Create sub block - Fully connected block
        self.sub_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=W*C,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(
                filters=W*C,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same')
        ])
        
        self.built = True

    def call(self, x, training=False, **kwargs):
        # sub-block 1
        x = self.sub_block1(x, training=training, **kwargs) + x
        x = self.dropout1(x, training=training, **kwargs)
        x = self.layer_norm1(x, training=training, **kwargs)
        
        # reshape
        x_shape = tf.shape(x)
        B, T = x_shape[0], x_shape[1]
        x = tf.reshape(x, shape=[B, T, 1, self.W * self.C])

        # sub-block 2
        x = self.sub_block2(x, training=training, **kwargs) + x
        x = self.dropout2(x)
        x = self.layer_norm2(x)
        
        # reshaping
        return tf.reshape(x, [B, T, self.W, self.C])
        
    def get_config(self):
        super_config = super(TimeDepthSeparableBlock, self).get_config()
        config = {
            'kernel_size': self.kernel_size,
            'dropout_prob': self.dropout_prob
        }
        return {**super_config, **config}
        