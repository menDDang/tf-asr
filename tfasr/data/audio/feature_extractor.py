from typing import Dict

import numpy as np
import tensorflow as tf

ONE_SECOND_IN_MS = 1000

SAMPLING_RATE_16K = int(16000)
DEFAULT_SAMPLING_RATE = SAMPLING_RATE_16K

DEFAULT_FRAME_LENGTH_MS = int(25)
DEFAULT_SLIDE_LENGTH_MS = int(10)


def _hann_offset_window_generator(window_length, dtype):
    """Computes a hanning window with offset.
    Args:
        window_length: The length of the window (typically frame size).
        dtype: TF data type
    Returns:
        Tensor of size frame_size with the window to apply.
    """
    arg = np.pi * 2.0 / (window_length)
    hann = 0.5 - (0.5 * np.cos(arg * (np.arange(window_length) + 0.5)))
    return hann.astype(dtype)


def _hann_window_generator(window_length, dtype):
    """Computes a standard version of Hann window.
    More details at https://en.wikipedia.org/wiki/Hann_function
    Args:
        window_length: The length of the window (typically frame size).
        dtype: TF data type
    Returns:
        Tensor of size frame_size with the window to apply.
    """
    arg = 2 * np.pi / window_length
    hann = 0.5 - 0.5 * np.cos(arg * np.arange(window_length))
    return hann.astype(dtype)


class FeatureExtractor(object):
    def __init__(self, 
                 feature_type: str,
                 bitrate=16, 
                 sample_rate=16000,
                 time_shift_ms=100,
                 preemph=0.97, 
                 frame_length_ms=25,
                 frame_step_ms=10,
                 window_type='hann',
                 fft_size=512,
                 use_squared_magnitude=False,
                 num_mel_bins=80,
                 num_mfcc_bins=40,
                 lower_edge_hertz=0.0,
                 upper_edge_hertz=8000,
                 top_db=80.0,
                 epsilon=1e-7,
                 seed=None,
                 training:bool=False):
        assert feature_type in ['magnitude', 'mel', 'logmel', 'mfcc']
        self.feature_type = feature_type
        self.bitrate = bitrate
        self.sample_rate = sample_rate
        self.time_shift_ms = time_shift_ms
        self.preemph = preemph
        self.frame_length_ms = frame_length_ms
        self.frame_step_ms = frame_step_ms
        self.window_type = window_type
        self.fft_size = fft_size
        self.use_squared_magnitude = use_squared_magnitude
        self.num_mel_bins = num_mel_bins
        self.num_mfcc_bins = num_mfcc_bins
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.top_db = top_db
        self.epsilon = epsilon
        self.seed = None
        self.training = training

        self.time_shift_length = int(self.time_shift_ms * self.sample_rate / 1000.0)
        self.frame_length = int(self.frame_length_ms * self.sample_rate / 1000.0)
        self.frame_step = int(self.frame_step_ms * self.sample_rate / 1000.0)        
        self.scale = tf.constant(pow(2, self.bitrate - 1), dtype=tf.float32)        
        #self.rand_time_shift = RandomTimeShift(time_shift_samples)
        #self.preemph = PreEmphasis(preemph=self.preemph)
        #self.windowing = Windowing(window_size=self.frame_length, window_type=self.window_type)

        if self.window_type == 'hann_offest':
            self.window = _hann_offset_window_generator(self.frame_length, np.float32)
        elif self.window_type == 'hann':
            self.window = _hann_window_generator(self.frame_length, np.float32)
        else:
            raise ValueError('unsupported window_type:%s' % self.window_type)

        num_spectrogram_bins = self.fft_size // 2 + 1
        self.mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz,
            dtype=tf.float32)

        self.max_length = None

    @property
    def nfft(self) -> int:
        return 2 ** (self.frame_length - 1).bit_length()

    @property
    def feature_dim(self) -> int:
        if self.feature_type == "spectrum":
            return self.nfft / 2 + 1
        elif self.feature_type == "mel" or self.feature_type == "logmel":
            return self.num_mel_bins
        elif self.feature_type == "mfcc":
            return self.num_mfcc_bins
        else:
            raise TypeError("invalid feature type")

    @property
    def shape(self) -> list:
        return [self.max_length, self.feature_dim]

    def update_length(self, new_length):
        if self.max_length is None:
            self.max_length = new_length
        else:
            self.max_length = max(self.max_length, new_length)
            
    def reset_length(self):
        self.max_length = None

    def extract(self, x):
        # cast integer encoded data into float & normalize
        x = tf.cast(x, dtype=tf.float32)
        x /= self.scale

        # apply time shift
        if self.training:
            x = self._rand_time_shift(x)

        # pre-emphasis signal
        x = self._preemph(x)

        # apply windowing (e.g. hanning)
        x = tf.signal.frame(x, frame_length=self.frame_length, frame_step=self.frame_step)
        x = x * self.window

        # short time Fourier transform & get magnitude
        x = tf.signal.rfft(x, [self.fft_size])
        x = tf.abs(x)
        if self.use_squared_magnitude:
            x = tf.square(x)
        else:
            x = x        
        if self.feature_type == "magnitude":
            return x

        if self.feature_type == "logspectrum":
            x = self._power_to_db(x)
            return x

        # mel spectrum
        x = tf.tensordot(x, self.mel_weight_matrix, axes=1)
        if self.feature_type == "mel":
            return x

        # logarize
        x = self._power_to_db(x)
        if self.feature_type == "logmel":
            return x

        # dicrete cosine transform (MFCC)
        x = tf.signal.mfccs_from_log_mel_spectrograms(x)
        x = tf.slice(x, begin=[0, 0], size=[-1, self.num_mfcc_bins])
        return x

    def _rand_time_shift(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Arguments:
            inputs: int/flaot encoded signals, shape of [T]
        """

        input_shape = tf.shape(inputs)
        time_length = tf.gather(input_shape, 0)

        time_shift_amounts = tf.random.uniform(
            shape=[1],
            minval=-self.time_shift_length,
            maxval=self.time_shift_length,
            dtype=tf.int32,
            seed=self.seed)
        amount = tf.gather(time_shift_amounts, 0)
        padding = tf.cond(
            amount > 0,
            lambda: [[amount, 0]],
            lambda: [[0, -amount]])
        outputs = tf.pad(
            inputs,
            paddings=padding,
            mode='CONSTANT')
        offset = tf.cond(
            amount > 0,
            lambda: [0],
            lambda: [-amount])
        outputs = tf.slice(outputs, begin=offset, size=[time_length])

        return outputs
        
    def _preemph(self, inputs):
        """
        Arguments:
            inputs: floating point encoded audio singal, shape of [T]
        """
        input_shape = tf.shape(inputs)
        time_length = tf.gather(input_shape, 0)

        s0 = tf.slice(inputs, begin=[0], size=[1])
        s1 = tf.slice(inputs, begin=[1], size=[time_length - 1])
        s1 -= self.preemph * tf.slice(inputs, begin=[0], size=[time_length - 1])

        return tf.concat([s0, s1], axis=-1)

    def _power_to_db(self, S: tf.Tensor) -> tf.Tensor:
        def log10(x):
            return tf.math.log(x) / tf.math.log(tf.constant(10, dtype=x.dtype))

        log_spec = 10.0 * log10(tf.maximum(S, self.epsilon))
        log_spec -= 10.0 * log10(tf.maximum(1.0, self.epsilon))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - self.top_db)
        return log_spec



class FeatureExtractor_old:

    def __init__(self, config: Dict):
        # signal
        self.sampling_rate = int(config.get("sampling_rate", DEFAULT_SAMPLING_RATE))
        frame_ms = int(config.get("frame_ms", DEFAULT_FRAME_LENGTH_MS))
        hop_ms = config.get("hop_ms", DEFAULT_SLIDE_LENGTH_MS)
        
        self.frame_length = int(
            self.sampling_rate * (float(frame_ms) / float(ONE_SECOND_IN_MS)))
        self.hop_length = int(
            self.sampling_rate * (float(hop_ms) / float(ONE_SECOND_IN_MS)))
        # for FFT
        self.center = bool(config.get("center", True))
        self.window_type = config.get("window_type", "hanning")
        self.preemphasis_coeff = float(config.get("preemphasis_coeff", 0.97))
        self.top_db = float(config.get("top_db", 80.0))
        # for feature
        self.num_mel_bins = int(config.get("num_mel_bins", 80))
        self.num_mfcc_bins = int(config.get("num_mfcc_bins", 40))
        self.min_hertz = float(config.get("min_hertz", 0.0))
        self.max_hertz = float(config.get("max_hertz", self.sampling_rate / 2))
        self.feature_type = config.get("feature_type", "logmel")
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=int(self.nfft / 2 + 1),
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.min_hertz,
            upper_edge_hertz=self.max_hertz
        )
        # for normalization
        self.normalize_signal = config.get("normalize_signal", True)
        self.normalize_feature = config.get("normalize_feature", True)
        # extra
        self.epsilon = float(config.get("epsilon", 1e-10))
        self.max_length = None
        
        assert self.window_type in ["hanning", "hamming"]
        assert self.feature_type in ["spectrum", "mel", "logmel", "mfcc"]
        
    @property
    def nfft(self) -> int:
        return 2 ** (self.frame_length - 1).bit_length()

    @property
    def feature_dim(self) -> int:
        if self.feature_type == "spectrum":
            return self.nfft / 2 + 1
        elif self.feature_type == "mel" or self.feature_type == "logmel":
            return self.num_mel_bins
        elif self.feature_type == "mfcc":
            return self.num_mfcc_bins
        else:
            raise TypeError("invalid feature type")

    @property
    def shape(self) -> list:
        return [self.max_length, self.feature_dim]

    @tf.function
    def update_length(self, new_length):
        if self.max_length is None:
            self.max_length = new_length
        else:
            self.max_length = max(self.max_length, new_length)
            
    def reset_length(self):
        self.max_length = None

    @tf.function
    def _stft(self, signal):
        if self.center:
            pad_size = self.nfft // 2
            signal = tf.pad(signal, [[pad_size, pad_size]], mode="REFLECT")
        
        if self.window_type == "hanning":
            window = tf.signal.hann_window(self.frame_length, periodic=True)
        elif self.window_type == "hamming":
            window = tf.signal.hamming_window(self.frame_length, periodic=True)
        else:
            raise TypeError("invalid window type")

        left_pad = (self.nfft - self.frame_length) // 2
        right_pad = self.nfft - self.frame_length - left_pad
        window = tf.pad(window, [[left_pad, right_pad]])
        framed_signals = tf.signal.frame(signal, frame_length=self.nfft, frame_step=self.hop_length)
        framed_signals *= window
        return tf.square(tf.abs(tf.signal.rfft(framed_signals, [self.nfft])))

    @tf.function
    def _power_to_db(self, S: tf.Tensor) -> tf.Tensor:
        def log10(x):
            return tf.math.log(x) / tf.math.log(tf.constant(10, dtype=x.dtype))

        log_spec = 10.0 * log10(tf.maximum(S, self.epsilon))
        log_spec -= 10.0 * log10(tf.maximum(1.0, self.epsilon))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - self.top_db)
        return log_spec

    @tf.function
    def extract(self, signal: tf.Tensor) -> tf.Tensor :
        # normalize signal
        if self.normalize_signal:
            gain = tf.reduce_max(
                tf.maximum(tf.abs(signal), self.epsilon),
                axis=-1)
            signal = signal / gain

        # pre-emphasis 
        if self.preemphasis_coeff is not None:
            s0 = tf.expand_dims(signal[0], axis=-1)
            s1 = signal[1:] - self.preemphasis_coeff * signal[:-1]
            signal = tf.concat([s0, s1], axis=-1)
        
        # Short Time Fourier Transform
        magnitude = self._stft(signal)
        if self.feature_type == "spectrum":
            spectrum = self._power_to_db(magnitude)
            return spectrum

        # Mel spectrum
        mel = tf.tensordot(magnitude, self.linear_to_mel_weight_matrix, 1)
        if self.feature_type == "mel":
            return mel

        # Log Mel spectrum
        log_mel = self._power_to_db(mel)
        if self.feature_type == "logmel":
            return log_mel

        # Mel Filter-bank Cepstrum Coeffeicients
        mfcc = tf.signal.mfccs_from_logmel_spectrograms(log_mel)
        return mfcc[:, :self.num_mfcc_bins]