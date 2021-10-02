import argparse

import tensorflow as tf

from .modules import *

class Seq2SeqTcr(tf.keras.Model):
    def __init__(self, hp, vocab_size, sos=2, eos=3, max_output_length=1000, **kwargs):
        super(Seq2SeqTcr, self).__init__(**kwargs)

        kernel_size = hp["tcr_net"]["kernel_size"]
        num_tcr_blocks = hp["tcr_net"]["num_tcr_blocks"]
        num_channels = hp["tcr_net"]["num_channels"]
        
        self.decoder_cell_units = num_channels
        
        attention_type = hp["attention_type"]

        self.vocab_size = vocab_size
        self.sos = sos
        self.eos = eos
        self.max_output_length = max_output_length

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(filters=num_channels,
                                   kernel_size=[kernel_size],
                                   strides=2,
                                   padding='same',
                                   activation='relu')
        ])
        for n in range(num_tcr_blocks):
            self.encoder.add(
                TemporalConvResidual(filters=num_channels, 
                                     kernel_size=kernel_size))

        if attention_type == "Bahdanau":
            self.attention = BahdanauAttention(num_channels)
        elif attention_type == "Dot":
            self.attention = InnerProductAttention(use_scale=False)
        elif attention_type == "ScaledDot":
            self.attention = InnerProductAttention(use_scale=True)
        else:
            raise ValueError("Invalid attention type")
            
        self.cell = tf.keras.layers.GRUCell(self.decoder_cell_units)
        self.fc = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        
    @tf.function
    def call(self, x, training=False, y_true=None, apply_soft_window=False):
        '''
        x: input tensor shape of [batch_size, input_time_length, num_mels],
        y_true: [B, output_time_length + 1], y_true[0] = {sos, y_1, y_2, ..., y_U, eos}
        '''
        # encode input tensor, x : [batch_size, input_time_length / 8, num_mels, C]
        x = self.encoder(x, training=training)

        B, T, D = x.shape
        if training:
            output_time_length = tf.shape(y_true)[1] - 1
            y_true = tf.cast(tf.one_hot(y_true, depth=self.vocab_size), tf.float32)
            query_u = self.cell.get_initial_state(batch_size=B, dtype=tf.float32)
            out = tf.TensorArray(tf.float32, output_time_length)
            for u in tf.range(output_time_length):
                _, query_u = self.cell(y_true[:, u, :], query_u)
                # attend : [B, D]
                attend = self.attention([query_u, x], 
                    apply_soft_window=apply_soft_window, 
                    max_output_length=output_time_length
                )
                y_pred_u = self.fc(tf.concat([query_u, attend], axis=1))
                out = out.write(u, y_pred_u)
            out = out.stack()
            out = tf.transpose(out, [1, 0, 2])
            return out
        else:
            y_pred_u = tf.zeros(shape=[B, self.vocab_size], dtype=tf.float32) + self.sos
            query_u = self.cell.get_initial_state(inputs=y_pred_u)
            out = tf.TensorArray(tf.float32, self.max_output_length)
            for u in tf.range(self.max_output_length):
                _, query_u = self.cell(y_pred_u, query_u)
                attend = self.attention([query_u, x])
                y_pred_u = self.fc(tf.concat([query_u, attend], axis=1))
                out = out.write(u, y_pred_u)
            out = out.stack()
            out = tf.transpose(out, [1, 0, 2])
            return out

