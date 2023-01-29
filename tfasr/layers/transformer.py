from __future__ import absolute_import
from typing import List, Dict, Union

import numpy as np
import tensorflow as tf

from tfasr.layers.positional_encoding import PositionalEncoding


class SpeechFeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 num_hiddens: int,
                 kernel_size: int = 11,
                 kernel_initializer: Union[str, tf.keras.initializers.Initializer] = 'glorot_uniform',
                 bias_initializer: Union[str, tf.keras.initializers.Initializer] = 'zeros',
                 kernel_regularizer: tf.keras.regularizers.Regularizer = None,
                 bias_regularizer: tf.keras.regularizers.Regularizer = None,
                 activity_regularizer: tf.keras.regularizers.Regularizer = None,
                 kernel_constraint: tf.keras.constraints.Constraint = None,
                 bias_constraint: tf.keras.constraints.Constraint = None,
                 **kwargs):
        super(SpeechFeatureEmbedding, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(
            filters=num_hiddens,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            data_format='channels_last',
            dilation_rate=1,
            groups=1,
            activation='relu',
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=f'{self.name}_conv1'
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=num_hiddens,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            data_format='channels_last',
            dilation_rate=1,
            groups=1,
            activation='relu',
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=f'{self.name}_conv2'
        )
        self.conv3 = tf.keras.layers.Conv1D(
            filters=num_hiddens,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            data_format='channels_last',
            dilation_rate=1,
            groups=1,
            activation='relu',
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=f'{self.name}_conv3'
        )

    def call(self, x, training: bool = False, **kwargs):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 depth: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 dropout_rate: float = 0.1,
                 kernel_initializer: Union[str, tf.keras.initializers.Initializer] = 'glorot_uniform',
                 bias_initializer: Union[str, tf.keras.initializers.Initializer] = 'zeros',
                 kernel_regularizer: tf.keras.regularizers.Regularizer = None,
                 bias_regularizer: tf.keras.regularizers.Regularizer = None,
                 activity_regularizer: tf.keras.regularizers.Regularizer = None,
                 kernel_constraint: tf.keras.constraints.Constraint = None,
                 bias_constraint: tf.keras.constraints.Constraint = None,
                 **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.feed_forward_dim = feed_forward_dim
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=depth,
            value_dim=depth,
            dropout=0.0,
            use_bias=True,
            output_shape=None,
            attention_axes=None,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        assert len(input_shape) == 3
        query_last_dim = input_shape[2]
        self.feed_forward_network = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=self.feed_forward_dim,
                    activation='relu',
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint),
                tf.keras.layers.Dense(
                    units=query_last_dim,
                    activation=None,
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint)
            ]
        )
        super().build(input_shape)

    def call(self, x, mask = None, training: bool = False, **kwargs):
        '''
        Args:
            x: shape of [batch_size, input_seq_len, d_model]
        '''
        attention_outputs = self.multi_head_attention(
            query=x,
            value=x,
            key=x,
            attention_mask=mask,
            return_attention_scores=False,
            training=training)
        attention_outputs = self.dropout1(
            attention_outputs, 
            training=training)
        # out1: (batch_size, input_seq_len, d_model)
        out1 = self.layer_norm1(
            tf.add(x, attention_outputs), 
            training=training) 

        ffn_outputs = self.feed_forward_network(out1) 
        ffn_outputs = self.dropout2(ffn_outputs, training=training)
        out2 = self.layer_norm2(
            tf.add(out1, ffn_outputs),
            training=training)
        return out2


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 speech_embed_dim: int,
                 num_encoder_layers: int,
                 depth: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 #input_vocab_size: int,
                 #max_pos_encoding: int,
                 dropout_rate: float = 0.1,
                 kernel_initializer: 
                    Union[
                        str, 
                        tf.keras.initializers.Initializer
                    ] = 'glorot_uniform',
                 bias_initializer: 
                    Union[
                        str, 
                        tf.keras.initializers.Initializer
                    ] = 'zeros',
                 kernel_regularizer: tf.keras.regularizers.Regularizer = None,
                 bias_regularizer: tf.keras.regularizers.Regularizer = None,
                 activity_regularizer: tf.keras.regularizers.Regularizer = None,
                 kernel_constraint: tf.keras.constraints.Constraint = None,
                 bias_constraint: tf.keras.constraints.Constraint = None,
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_encoder_layers = num_encoder_layers

        self.speech_embedding = SpeechFeatureEmbedding(
            num_hiddens=speech_embed_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name='speech_embedding'
        )
        self.positional_embeddings = PositionalEncoding(
            alpha=1, 
            beta=0, 
            name=f'{self.name}_PositionalEmbeddings')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoders = [
            TransformerEncoderLayer(
                depth=depth,
                num_heads=num_heads,
                feed_forward_dim=feed_forward_dim,
                dropout_rate=dropout_rate,                    
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                name=f'TransformerEncoderLayer_{n}')
            for n in range(self.num_encoder_layers)
        ]
    
    def call(self,
             x: Union[np.ndarray, tf.Tensor],
             mask: Union[np.ndarray, tf.Tensor] = None,
             training: bool = False,
             **kwargs):
        x = self.speech_embedding(x, training=training)
        x = tf.add(self.positional_embeddings(x, training=training), x)
        x = self.dropout(x, training=training)

        for n in range(self.num_encoder_layers):
            x = self.encoders[n](x, mask=mask, training=training)
            
        return x