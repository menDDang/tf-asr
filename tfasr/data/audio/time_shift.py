from __future__ import absolute_import

import tensorflow as tf


class RandTimeShift(tf.keras.layers.Layer):
    def __init__(self, 
                 time_shift_length: int,
                 seed=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.time_shift_length = time_shift_length
        self.seed = seed
    
    def build(self, input_shape):
        self.batch_size = input_shape[0]
        super().build(input_shape)

    def call(self, x, training: bool = False):

        def _shift(_x):
            time_length = tf.gather_nd(tf.shape(_x), [1])            
            time_shift_amounts = tf.random.uniform(
                shape=[self.batch_size],
                minval=-self.time_shift_length,
                maxval=self.time_shift_length,
                dtype=tf.int32,
                seed=self.seed)
            
            def _cond(_b, _outputs):
                return tf.less(_b, self.batch_size)

            def _body(_b, _outputs):
                amount = tf.gather_nd(time_shift_amounts, [_b])
                padding = tf.cond(
                    tf.greater(amount, 0),
                    lambda: tf.concat([[amount], [0]], axis=0),
                    lambda: tf.concat([[0], [-amount]], axis=0))
                padding = tf.expand_dims(padding, axis=0)
                _x_b = tf.slice(_x, begin=[_b, 0], size=[1, -1])
                _x_b = tf.squeeze(_x_b, axis=0)
                _x_b = tf.pad(_x_b, paddings=padding, mode='CONSTANT', constant_values=0)
                offset = tf.cond(
                    tf.greater(amount, 0),
                    lambda: [0],
                    lambda: [-amount])
                _x_b = tf.slice(_x_b, begin=offset, size=[time_length])
                return tf.add(_b, 1), _outputs.write(_b, _x_b)

            b = tf.constant(0, dtype=tf.int32)
            outputs = tf.TensorArray(
                dtype=tf.float32,
                size=self.batch_size,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([time_length]))
            _, outputs = tf.while_loop(
                cond=_cond,
                body=_body,
                loop_vars=[b, outputs])
            outputs = outputs.stack()
            return outputs

        x = tf.cond(
            tf.equal(training, True),
            lambda: _shift(x),
            lambda: x
        )
        return x