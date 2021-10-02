import tensorflow as tf

class TemporalConvResidual(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, **kwargs):
        super(TemporalConvResidual, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size

        self.nn = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters, 
                                   kernel_size=[kernel_size], 
                                   strides=1, 
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(filters, 
                                   kernel_size=[kernel_size], 
                                   strides=1, 
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.BatchNormalization()
        ])

    @tf.function
    def call(self, x, training=False, **kwargs):
        x = self.nn(x, training=training) + x
        x = tf.nn.relu(x)
        return x 