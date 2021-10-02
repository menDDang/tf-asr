import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        
        self.W_s = tf.keras.layers.Dense(units, use_bias=False)
        self.W_h = tf.keras.layers.Dense(units, use_bias=False)
        self.W = tf.keras.layers.Dense(1, use_bias=False, activation='softmax')

    @tf.function
    def call(self, x):
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

        query = tf.expand_dims(query, axis=1)
        attend = self.W(tf.nn.tanh(self.W_s(query) + self.W_h(keys)))
        context = tf.reduce_sum(values * attend, axis=1)
        return context

if __name__ == "__main__":
    batch_size = 3
    input_time_length = 4
    input_dim = 5
    query_dim = 6

    h = tf.zeros(shape=[batch_size, input_time_length, input_dim], dtype=tf.float32)
    query_before = tf.zeros(shape=[batch_size, query_dim], dtype=tf.float32)
    attention = BahdanauAttention(7)

    score = attention([query_before, h])
    print(score.shape)
