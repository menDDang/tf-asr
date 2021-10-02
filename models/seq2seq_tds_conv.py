import argparse

import tensorflow as tf

from .modules import BahdanauAttention
from .modules import DotProductAttention
from .modules.time_depth_seperable_conv import TimeDepthSeparableConv2D


class Seq2SeqTdsConv(tf.keras.Model):
    def __init__(self, hp, vocab_size, sos=2, eos=3, max_output_length=1000, **kwargs):
        super(Seq2SeqTdsConv, self).__init__(**kwargs)

        num_mels = hp["num_mels"]
        kernel_size = hp["tds_conv_net"]["kernel_size"]
        num_tds_blocks_1 = hp["tds_conv_net"]["num_tds_blocks_1"]
        num_channels_1 = hp["tds_conv_net"]["num_channels_1"]
        num_tds_blocks_2 = hp["tds_conv_net"]["num_tds_blocks_2"]
        num_channels_2 = hp["tds_conv_net"]["num_channels_2"]
        num_tds_blocks_3 = hp["tds_conv_net"]["num_tds_blocks_3"]
        num_channels_3 = hp["tds_conv_net"]["num_channels_3"]
        encoder_dense_units = hp["tds_conv_net"]["encoder_dense_units"]
        
        self.decoder_cell_units = encoder_dense_units
        
        attention_unit_num = hp["attention_unit_num"]
        attention_type = hp["attention_type"]

        self.vocab_size = vocab_size
        self.sos = sos
        self.eos = eos
        self.max_output_length = max_output_length

        self.encoder = tf.keras.Sequential()
        # first convolution layer
        self.encoder.add(
            tf.keras.layers.Conv2D(filters=num_channels_1,
                                   kernel_size=[kernel_size, 1],
                                   strides=[2, 1],
                                   padding='same',
                                   activation='relu'
        ))
        self.encoder.add(tf.keras.layers.Dropout(0.2))
        # first TDS blocks
        for n in range(num_tds_blocks_1):
            self.encoder.add(
                TimeDepthSeparableConv2D(filters=num_channels_1,
                                         kernel_size=kernel_size
            ))
        # second convolution layer
        self.encoder.add(
            tf.keras.layers.Conv2D(filters=num_channels_2,
                                   kernel_size=[kernel_size, 1],
                                   strides=[2, 1],
                                   padding='same',
                                   activation='relu'
        ))
        self.encoder.add(tf.keras.layers.Dropout(0.2))
        # second TDS blocks
        for n in range(num_tds_blocks_2):
            self.encoder.add(
                TimeDepthSeparableConv2D(filters=num_channels_2,
                                         kernel_size=kernel_size
            ))
        # Last convoultion layer
        self.encoder.add(
            tf.keras.layers.Conv2D(filters=num_channels_3,
                                   kernel_size=[kernel_size, 1],
                                   strides=[2, 1],
                                   padding='same',
                                   activation='relu'
        ))
        self.encoder.add(tf.keras.layers.Dropout(0.2))
        # Last TDS blocks
        for n in range(num_tds_blocks_3):
            self.encoder.add(
                TimeDepthSeparableConv2D(filters=num_channels_3,
                                         kernel_size=kernel_size
            ))
        # Fully connected layer
        self.encoder_fc = tf.keras.layers.Dense(encoder_dense_units)
        
        if attention_type == "Bahdanau":
            self.attention = BahdanauAttention(attention_unit_num)
        elif attention_type == "Dot":
            self.attention = DotProductAttention(use_scale=False)
        elif attention_type == "ScaledDot":
            self.attention = DotProductAttention(use_scale=True)

        self.decoder_fc1 = tf.keras.layers.Dense(encoder_dense_units)
        self.decoder_cell = tf.keras.layers.GRUCell(self.decoder_cell_units)
        self.fc = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        
    @tf.function
    def call(self, x, training=False, y_true=None, apply_soft_window=False):
        '''
        x: input tensor shape of [batch_size, input_time_length, num_mels],
        y_true: [B, output_time_length + 1], y_true[0] = {sos, y_1, y_2, ..., y_U, eos}
        '''
        # expand dimension, x : [batch_size, input_time_length, num_mels, 1]
        x = tf.expand_dims(x, axis=-1)

        # encode input tensor, x : [batch_size, input_time_length / 8, num_mels, C]
        x = self.encoder(x, training=training)

        B, T, D, C = x.shape
        x = tf.reshape(x, [B, T, D * C])
        x = self.encoder_fc(x)
        
        if training:
            y_true = tf.cast(tf.one_hot(y_true, depth=self.vocab_size), tf.float32)
        
        states = self.decoder_cell.get_initial_state(batch_size=B, dtype=tf.float32)
        y_pred_before = tf.zeros(shape=[B, self.vocab_size], dtype=tf.float32) + self.sos
        y_pred = tf.TensorArray(dtype=tf.float32, size=self.max_output_length)
        for u in tf.range(self.max_output_length):
            context = self.attention([states, x])

            cell_inputs = tf.concat([y_pred_before, context], axis=1)
            y_pred_now, states = self.decoder_cell(cell_inputs, states)
            y_pred_now = self.fc(y_pred_now)
            y_pred = y_pred.write(u, y_pred_now)

        y_pred = y_pred.stack()
        y_pred = tf.transpose(y_pred, [1, 0, 2])
        return y_pred
        