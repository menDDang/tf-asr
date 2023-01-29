from __future__ import absolute_import
from typing import Dict, List

import tensorflow as tf

from tfasr.layers.spec_augment import SpecAugment
from tfasr.layers.time_depth_seperable import TimeDepthSeparableBlock
from tfasr.models.transducer_base import TransducerBase
from tfasr import utils


class TDSEncoder(tf.keras.layers.Layer):
    def __init__(self, 
                 kernel_size: int, 
                 num_tds_blocks: List[int],
                 num_tds_channels: List[int],
                 dropout_prob:float = 0.1,
                 **kwargs):
        super(TDSEncoder, self).__init__(**kwargs)

        assert len(num_tds_blocks) == len(num_tds_channels)
        if dropout_prob is not None:
            assert (0 < dropout_prob < 1)

        self.kernel_size = kernel_size
        self.num_tds_blocks = num_tds_blocks
        self.num_tds_channels = num_tds_channels
        self.dropout_prob = dropout_prob

        self.last_tds_channel = self.num_tds_channels[-1]
        self.nn = tf.keras.Sequential()
        for n in range(len(num_tds_blocks)):
            # convolution layer
            num_blocks = num_tds_blocks[n]
            num_channels = num_tds_channels[n]
            self.nn.add(
                tf.keras.layers.Conv2D(
                    filters=num_channels,
                    kernel_size=self.kernel_size,
                    strides=[2, 1],
                    padding='same',
                    activation='relu'))
            # TDS blocks
            for n in range(num_blocks):
                self.nn.add(
                    TimeDepthSeparableBlock(
                        kernel_size=self.kernel_size,
                        dropout_prob=self.dropout_prob))
        

    def build(self, input_shape):
        super(TDSEncoder, self).build(input_shape)
        _, _, self.W = input_shape
        
    def call(self, x, **kwargs):
        x = tf.expand_dims(x, axis=-1)
        x = self.nn(x, **kwargs)
        x_shape = tf.shape(x)
        T = tf.gather(x_shape, 1)
        x = tf.reshape(x, shape=[-1, T, self.W * self.last_tds_channel])
        return x

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_tds_blocks': self.num_tds_blocks,
            'num_tds_channels': self.num_tds_channels,
            'dropout_prob': self.dropout_prob
        }
        base_config = super(TDSEncoder, self).get_config()
        return {**base_config, **config}
        

      
class TDSDecoder(tf.keras.layers.Layer):
    def __init__(self, 
                 attention_type: str,
                 attention_dim: int,
                 num_units: int,
                 vocab_size: int):
        super(TDSDecoder, self).__init__()
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.num_units = num_units
        self.vocab_size = vocab_size

        self.proj_query = tf.keras.layers.Dense(self.attention_dim)
        self.proj_key = tf.keras.layers.Dense(self.attention_dim)

        # Set attention
        if self.attention_type == "dot":
            self.attention = tf.keras.layers.Attention(use_scale=False)
        elif self.attention_type == "scaled_dot":
            self.attention = tf.keras.layers.Attention(use_scale=True)
        else:
            raise ValueError("Invalid attention type")
        
        # Set RNN
        self.rnn1 = tf.keras.layers.GRU(self.num_units, return_sequences=True)        
        self.dense = tf.keras.layers.Dense(self.vocab_size, activation="softmax")

    def call(self, inputs, **kwargs):
        x = inputs["inputs"]
        predictions = inputs["predictions"]
        predictions = tf.one_hot(
            predictions, 
            depth=self.vocab_size,
            dtype=tf.float32)
        query = self.proj_query(predictions, **kwargs)
        key = self.proj_key(x, **kwargs)

        query_mask = tf.sequence_mask(
            inputs["prediction_lengths"],
            maxlen=tf.gather(tf.shape(query), 1))
        value_mask = tf.sequence_mask(
            inputs["input_lengths"],
            maxlen=tf.gather(tf.shape(key), 1))
        attention_context = self.attention(
            inputs=[query, key], 
            mask=[query_mask, value_mask],
            **kwargs)
                
        out = self.rnn1(attention_context, **kwargs)
        out = self.dense(out, **kwargs)
        return out

class TDS(TransducerBase):
    def __init__(self,
                 encoder_kernel_size: int,
                 encoder_num_tds_blocks: List[int],
                 encoder_num_tds_channels: List[int],
                 decoder_num_units: int,
                 decoder_attention_type: str,
                 decoder_attention_dim: int,
                 vocab_size: int,
                 encoder_dropout_prob: float = None,
                 **kwargs):
        super(TDS, self).__init__(
            encoder=TDSEncoder(kernel_size=encoder_kernel_size,
                               num_tds_blocks=encoder_num_tds_blocks,
                               num_tds_channels=encoder_num_tds_channels,
                               dropout_prob=encoder_dropout_prob),
            decoder=TDSDecoder(attention_type=decoder_attention_type,
                               attention_dim=decoder_attention_dim,
                               num_units=decoder_num_units,
                               vocab_size=vocab_size),
            **kwargs
        )
        self.time_reduction_factor = 2 ** len(encoder_num_tds_blocks)
    
