from __future__ import absolute_import

import tensorflow as tf


class MFCC(tf.keras.layers.Layer):
    def __init__(self, 
                 sample_rate: int,
                 fft_size: int,
                 num_mel_bins: int,
                 num_mfcc_bins: int,
                 lower_edge_hertz: float = None,
                 upper_edge_hertz: float = None,
                 top_db: float = 80.0,
                 epsilon: float = 1e-10,
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
        self.num_mfcc_bins = num_mfcc_bins
        self.top_db = top_db
        self.epsilon = epsilon

    def call(self, x: tf.Tensor) -> tf.Tensor:
        '''
        Args:
            x: magnitudes of input signal, shape of [B, T, D]
        '''
        
        def log10(_x):
            return tf.math.divide(
                tf.math.log(_x),
                tf.math.log(tf.constant(10.0, dtype=_x.dtype)))

        def _power_to_db(_x):
            _x = tf.math.multiply(
                10.0, 
                log10(tf.math.maximum(_x, self.epsilon)))                
            _x = tf.maximum(
                _x,
                tf.subtract(
                    tf.reduce_max(_x),
                    self.top_db
                )
            )
            return _x

        x = tf.tensordot(x, self.mel_matrix, axes=1)
        x = _power_to_db(x)

        # dicrete cosine transform (MFCC)
        x = tf.signal.mfccs_from_log_mel_spectrograms(x)
        x = tf.slice(x, begin=[0, 0, 0], size=[-1, -1, self.num_mfcc_bins])
        return x