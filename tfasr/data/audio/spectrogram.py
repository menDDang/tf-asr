from __future__ import absolute_import

import tensorflow as tf

from .time_shift import RandTimeShift
from .pre_emphasis import PreEmphasis


class Spectrogram(tf.keras.layers.Layer):
    def __init__(self, 
                 frame_length: int,
                 frame_step: int,
                 window_type: str = 'hanning',
                 fft_size: int = 512,
                 **kwargs):
        super().__init__(**kwargs)
        assert window_type in ['hanning', 'hamming']
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_size = fft_size
        
        if window_type == 'hanning':
            self.window = tf.signal.hann_window(
                self.frame_length,
                periodic=True,
                dtype=tf.float32,
                name='hanning_window')
        elif window_type == 'hamming':
            self.window = tf.signal.hamming_window(
                self.frame_length,
                periodic=True,
                dtype=tf.float32,
                name='hamming_window')

            
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.signal.frame(
            x, 
            frame_length=self.frame_length, 
            frame_step=self.frame_step)
        x = tf.multiply(x, self.window)
        x = tf.signal.rfft(x, fft_length=[self.fft_size])
        x = tf.abs(x)
        x = tf.square(x)
        return x