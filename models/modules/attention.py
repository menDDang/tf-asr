import numpy as np
import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)


    def call(self, x):
        # query : hidden states from before step. shape of [batch_size, hidden_state_dim]
        # value : input tensor. shape of [batch_size, time_length, input_dim]
        query, values = x
        
        # Expand dimension of query
        # [batch_size, hidden_state_dim] -> [batch_size, 1, hidden_state_dim]
        query_with_time_axis = tf.expand_dims(query, axis=1)

        # Compute score : [batch, time_length, 1]
        score = tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values))
        score = self.V(score)
        
        # Compute attention weights : [batch_size, time_length, 1]
        attention_weights = tf.nn.softmax(score, axis=1)

        # Compute context : [batch_size, time_length, hidden_state_dim]
        context = attention_weights * values

        # Add context along time axis
        context = tf.reduce_sum(context, axis=1)
        
        return context, attention_weights



class InnerProductAttention(tf.keras.layers.Layer):
    def __init__(self, use_scale=False, sigma=4):
        super(InnerProductAttention, self).__init__()

        self.use_scale = use_scale
        self.sigma = sigma


    @tf.function
    def call(self, x, apply_soft_window=False, max_output_length=None):
        '''
        x : [query, values]
        query : hidden states from before step. shape of [batch_size, input_dim]
        values : input tensor. shape of [batch_size, input_time_length, input_dim]

        shape of output : [batch_size, input_dim]
        '''
        
        query, values = x
        _, T, D = values.shape

        if len(query.shape) == 2:
            score = tf.linalg.matmul(values, tf.expand_dims(query, axis=1), transpose_b=True)
            if self.use_scale:
                score = score / tf.sqrt(tf.constant(D, dtype=tf.float32))
            
            if apply_soft_window:
                U = max_output_length
                W = np.zeros(shape=[T, U], dtype=np.float32)
                for i in range(T):
                    for j in range(U):
                        W[i, j] = np.power(i - float(T) / float(U) * j, 2)
                W = tf.constant(W, dtype=tf.float32) / (2 * self.sigma * self.sigma)
                score = score - W
            
            score = tf.nn.softmax(score, axis=2)
            context = tf.reduce_sum(score * values, axis=1)
            
            return context
        else:
            _, OT, D = query.shape
            scale = 1 / tf.sqrt(tf.constant(D, dtype=tf.float32)) if self.use_scale else 1
            context = tf.stack([
                tf.reduce_sum(tf.nn.softmax(
                    tf.linalg.matmul(values, tf.expand_dims(query[:, ot, :], axis=1), transpose_b=True) * scale,
                    axis=2
                ) * values, axis=1)
                for ot in range(OT)
            ], axis=1)
            return context