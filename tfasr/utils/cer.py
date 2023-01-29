from __future__ import absolute_import

import tensorflow as tf


class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """
    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def compute_cer(self, y_true, y_pred):
        hypothesis = tf.argmax(y_pred["logits"], axis=-1, output_type=tf.int32)
        hypothesis_lengths = y_pred["logit_lengths"]

        truth = y_true["labels"]
        truth_lengths = y_true["label_lengths"]

        batch_size = tf.gather(tf.shape(hypothesis), 0)

        def condition(_b, _):
            return tf.less(_b, batch_size)

        def body(_b, _outputs):
            h_length = tf.gather(hypothesis_lengths, _b)
            h = tf.slice(hypothesis, begin=[_b, 0], size=[1, h_length-1])
            t_length = tf.gather(truth_lengths, _b)
            t = tf.slice(truth, begin=[_b, 0], size=[1, t_length-1])
            distance = tf.edit_distance(
                hypothesis=tf.sparse.from_dense(h),
                truth=tf.sparse.from_dense(t),
                normalize=True)
            return (tf.add(_b, 1), _outputs.write(_b, distance))

        b = tf.constant(0, dtype=tf.int32)
        distances = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=False, element_shape=[1])
        b, distances = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[b, distances])
        return distances.stack()

    def update_state(self, y_true, y_pred):
        distance = self.compute_cer(y_true, y_pred)        
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))

        batch_size = tf.gather(tf.shape(y_true["labels"]), 0)
        self.counter.assign_add(tf.cast(batch_size, tf.float32))
        
    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_state(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)

