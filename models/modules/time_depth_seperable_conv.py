import tensorflow as tf

class TimeDepthSeparableConv2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, **kwargs):
        super(TimeDepthSeparableConv2D, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv1 = tf.keras.layers.Conv2D(
            filters, 
            kernel_size=[kernel_size, 1], 
            strides=[1, 1], 
            padding='same',
            activation='relu'
        )
        self.layer_norm1 = tf.keras.layers.LayerNormalization()

        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same',
                activation='relu'),
            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same')
        ])
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, x, **kwargs):
        # 2D convolution over time
        # Fig 1. (b)
        conv1_out = self.conv1(x)
        layer_norm1_out = self.layer_norm1(conv1_out + x)
        
        # fully connected block
        # Fig 1. (c)
        _, T, W, C = layer_norm1_out.shape
        x = tf.reshape(layer_norm1_out, [-1, T, 1, W * C])
        conv2_out = self.conv2(x)
        layer_norm2_out = self.layer_norm2(conv2_out + x)
        
        return tf.reshape(layer_norm2_out, [-1, T, W, C])