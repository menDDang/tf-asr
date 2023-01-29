from __future__ import absolute_import

import tensorflow as tf

from .time_shift import RandTimeShift
from .pre_emphasis import PreEmphasis
from .spectrogram import Spectrogram
from .mel_spectrogram import MelSpectrogram
from .log_mel_spectrogram import LogMelSpectrogram
from .mfcc import MFCC

class FeatureExtractor(tf.keras.layers.Layer):
    
    def __init__(self, 
                 feature_type: str,
                 sample_rate=16000,
                 time_shift_ms=100,
                 preemph=0.97, 
                 frame_length_ms=25,
                 frame_step_ms=10,
                 window_type: str = 'hanning',
                 fft_size: int = 512,
                 num_mel_bins: int = 80,
                 num_mfcc_bins: int = 40,
                 lower_edge_hertz: float = None,
                 upper_edge_hertz: float = 8000,
                 top_db: float = 80.0,
                 epsilon: float = 1e-7,
                 seed=None,
                 **kwargs):
        assert feature_type in ['spectrogram', 'mel', 'logmel', 'mfcc']
        super().__init__(**kwargs)
        self.feature_type = feature_type
           
        self.top_db = top_db
        self.epsilon = epsilon        

        routines = [
            RandTimeShift(
                time_shift_length=int(time_shift_ms * sample_rate / 1000.0),
                seed=seed),
            PreEmphasis(preemph=preemph),
            Spectrogram(
                frame_length=int(frame_length_ms * sample_rate / 1000.0),
                frame_step=int(frame_step_ms * sample_rate / 1000.0),
                window_type=window_type,
                fft_size=fft_size)
        ]

        if feature_type == 'mel':
            routines += [
                MelSpectrogram(
                    sample_rate=sample_rate,
                    fft_size=fft_size,
                    num_mel_bins=num_mel_bins,
                    lower_edge_hertz=lower_edge_hertz,
                    upper_edge_hertz=upper_edge_hertz)
            ]
        elif feature_type == 'logmel':
            routines += [
                LogMelSpectrogram(
                    sample_rate=sample_rate,
                    fft_size=fft_size,
                    num_mel_bins=num_mel_bins,
                    lower_edge_hertz=lower_edge_hertz,
                    upper_edge_hertz=upper_edge_hertz,
                    top_db=top_db,
                    epsilon=epsilon)
            ]
        elif feature_type == 'mfcc':
            routines += [
                MFCC(
                    sample_rate=sample_rate,
                    fft_size=fft_size,
                    num_mel_bins=num_mel_bins,
                    num_mfcc_bins=num_mfcc_bins,
                    lower_edge_hertz=lower_edge_hertz,
                    upper_edge_hertz=upper_edge_hertz,
                    top_db=top_db,
                    epsilon=epsilon)
            ]

        self.routines = tf.keras.Sequential(routines)
            
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:        
        x = self.routines(x, training=training)

        return x