from __future__ import absolute_import

import tensorflow as tf


class MelSpectrogram(tf.keras.layers.Layer):
    def __init__(self, 
                 sample_rate: int,
                 fft_size: int,
                 num_mel_bins: int,
                 lower_edge_hertz: float = None,
                 upper_edge_hertz: float = None,
                 **kwargs):
        super().__init__(**kwargs)
            
        self.mel_matrix = tf.Variable(
            initial_value=tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=num_mel_bins,
                num_spectrogram_bins=int(fft_size // 2 + 1),
                sample_rate=sample_rate,
                lower_edge_hertz=lower_edge_hertz if lower_edge_hertz is not None else 0.0,
                upper_edge_hertz=upper_edge_hertz if upper_edge_hertz is not None else float(sample_rate / 2),
                dtype=tf.float32),
            trainable=False,
            name='mel_matrix',
            dtype=tf.float32)
            
    def call(self, x: tf.Tensor) -> tf.Tensor:
        '''
        Args:
            x: magnitudes of input signal, shape of [B, T, D]
        '''
        x = tf.tensordot(x, self.mel_matrix, axes=1)
        return x
