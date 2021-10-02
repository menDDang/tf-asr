import argparse
from json import decoder

import tensorflow as tf

#from .modules.attention import BahdanauAttention, InnerProductAttention
from .modules import BahdanauAttention
from .modules import DotProductAttention

class Seq2SeqLstm(tf.keras.Model):
    def __init__(self, hp, vocab_size, sos=1, eos=2, max_output_length=1000, **kwargs):

        super(Seq2SeqLstm, self).__init__(**kwargs)

        encoder_conv_filters = hp["lstm_net"]["encoder_conv_filters"]
        encoder_conv_kernel_size = hp["lstm_net"]["encoder_conv_kernel_size"]

        self.encoder_lstm_num_layers = hp["lstm_net"]["encoder_lstm_num_layers"]
        encoder_lstm_units = hp["lstm_net"]["encoder_lstm_units"]
        #encoder_dropout_prob = hp["lstm_net"]["encoder_dropout_prob"]

        attention_unit_num = hp["attention_unit_num"]
        attention_type = hp["attention_type"]

        decoder_lstm_units = hp["lstm_net"]["decoder_lstm_units"]

        self.vocab_size = vocab_size
        self.sos = sos
        self.eos = eos
        self.max_output_length = max_output_length

        self.encoder_conv = tf.keras.layers.Conv1D(
            filters=encoder_conv_filters,
            kernel_size=encoder_conv_kernel_size,
            strides=2,
            activation='relu'
        )

        self.encoder_layers = []
        for n in range(self.encoder_lstm_num_layers):
            self.encoder_layers.append(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        encoder_lstm_units, 
                        return_sequences=True,
                        return_state=True)))

        if attention_type == "Bahdanau":
            self.attention = BahdanauAttention(attention_unit_num)
        elif attention_type == "Dot":
            self.attention = DotProductAttention(use_scale=False)
        elif attention_type == "ScaledDot":
            self.attention = DotProductAttention(use_scale=True)

        self.decoder_cell = tf.keras.layers.LSTMCell(decoder_lstm_units)
        
        self.fc = tf.keras.Sequential()
        self.fc.add(tf.keras.layers.Dense(512, activation='relu'))
        self.fc.add(tf.keras.layers.Dense(256, activation='relu'))
        self.fc.add(tf.keras.layers.Dense(64, activation='relu'))
        self.fc.add(tf.keras.layers.Dense(self.vocab_size, activation='softmax'))

    @tf.function
    def call(self, x, training=False, y_true=None, apply_soft_window=False):
        B, T, D = x.shape
        #x = tf.reshape(x, [B, int(T / 3), D * 3])
        x = self.encoder_conv(x)
        
        for n in range(self.encoder_lstm_num_layers):
            x, states_0, states_1, states_2, states_3 = self.encoder_layers[n](x)
        encoder_outputs = x
        states = [
            tf.concat([states_0, states_2], axis=1), 
            tf.concat([states_1, states_3], axis=1)]
        #states = self.decoder_cell.get_initial_state(batch_size=B, dtype=tf.float32)
            
        y_pred_before = tf.zeros(shape=[B], dtype=tf.int32) + self.sos
        y_pred_before = tf.cast(
            tf.one_hot(y_pred_before, depth=self.vocab_size),
            tf.float32
        )

        if training:
            output_time_length = y_true.shape[1]
        else:
            output_time_length = self.max_output_length
        y_pred = tf.TensorArray(tf.float32, size=output_time_length)
        if training:
            y_true = tf.cast(tf.one_hot(y_true, depth=self.vocab_size), tf.float32)            

        for t in tf.range(output_time_length):
            # Apply attention
            # shape of outputs : [batch_size, attention_unit], [batch_size, time_length, 1]
            context = self.attention([states[0], encoder_outputs])
            
            cell_inputs = tf.concat([y_pred_before, context], axis=1)
            y_pred_now, states = self.decoder_cell(cell_inputs, states)
            y_pred_now = self.fc(y_pred_now)
            y_pred = y_pred.write(t, y_pred_now)

            if training:
                y_pred_before = y_true[:, t]
            else:
                y_pred_before = y_pred_now

        y_pred = y_pred.stack()
        y_pred = tf.transpose(y_pred, [1, 0, 2])

        return y_pred

