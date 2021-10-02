import numpy as np
import tensorflow as tf

class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self, use_scale=False, max_input_length=None, max_output_length=None, sigma=4):
        super(DotProductAttention, self).__init__()

        self.use_scale = use_scale
        
        if max_input_length is not None and max_output_length is not None:
            W = np.zeros(shape=[max_input_length, max_output_length], dtype=np.float32)
            for i in range(max_input_length):
                for j in range(max_output_length):
                    W[i, j] = np.power(i - float(max_input_length) / float(max_output_length) * j, 2)
            self.W = tf.constant(W, dtype=tf.float32) / (2 * sigma * sigma)
        else:
            self.W = None

    @tf.function
    def call(self, x, apply_soft_window=False):
        # B : batch_size
        # U : output_time_length
        # T : input_time_length

        # query : hidden states from before step, [B, hidden_state_dim]
        # value : input tensor, [B, T, input_dim]
        if len(x) == 3:
            query, keys, values = x
        else:
            query, keys = x
            values = keys

        B, T, D = keys.shape

        query = tf.expand_dims(query, axis=1)
        attend = tf.linalg.matmul(keys, query, transpose_b=True)
        if self.use_scale:
            attend = attend / tf.sqrt(tf.constant(D, dtype=tf.float32))
        if apply_soft_window and self.W is not None:
            attend = attend - self.W
        attend = tf.nn.softmax(attend, axis=2)
        context = tf.reduce_sum(values * attend, axis=1)
        return context


if __name__ == "__main__":
    batch_size = 3
    input_time_length = 4
    input_dim = 5
    query_dim = input_dim

    h = tf.zeros(shape=[batch_size, input_time_length, input_dim], dtype=tf.float32)
    query_before = tf.zeros(shape=[batch_size, query_dim], dtype=tf.float32)
    attention = DotProductAttention()

    score = attention([query_before, h])
    print(score.shape)