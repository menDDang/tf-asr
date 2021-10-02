import abc
import math

import soundfile as sf
import tensorflow as tf

from .dataset_base import BaseDataset, BUFFER_SIZE

class AsrDataset(BaseDataset):
    def __init__(
        self,
        config : dict = None
    ):
        super(AsrDataset, self).__init__()
        if config is None:
            config = dict()
        self.frame_length_ms = config.get("frame_length_ms", 25)
        self.hop_length_ms = config.get("hop_length_ms", 10)
        self.nfft = config.get("nfft", 512)
        
        # Samples
        self.sample_rate = config.get("sample_rate", 16000)
        self.frame_length = int(self.sample_rate * (config.get("frame_ms", 25) / 1000))
        self.frame_step = int(self.sample_rate * (config.get("stride_ms", 10) / 1000))
        # Features
        self.num_feature_bins = config.get("num_feature_bins", 80)
        self.feature_type = config.get("feature_type", "log_mel_spectrogram")
        self.preemphasis_coeff = config.get("preemphasis", None)
        self.top_db = config.get("top_db", 80.0)
        # Normalization
        self.normalize_signal = config.get("normalize_signal", True)
        self.normalize_feature = config.get("normalize_feature", True)
        self.normalize_per_frame = config.get("normalize_per_frame", False)
        self.center = config.get("fft_center", True)

    @abc.abstractmethod
    def parse(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, file_path):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, file_path):
        raise NotImplementedError()


    def read_audio(self, file_path):
        audio, sr = sf.read(file_path)
        return audio, sr

    def normalize_feature_fn(
        self,
        audio_feature : tf.Tensor,
        per_frame=False) ->tf.Tensor:
        """
        Mean and variance features normalization
        Args:
            audio_feature: tf.Tensor with shape [T, F]

        Returns:
            normalized audio features with shape [T, F]
        """

        axis = 1 if per_frame else None
        mean = tf.reduce_mean(audio_feature, axis=axis, keepdims=True)
        var = tf.math.reduce_variance(audio_feature, axis=axis, keepdims=True)
        return (audio_feature - mean) / tf.math.sqrt(tf.maximum(var, 1e-9))


    def normalize_audio(
        self,
        signal: tf.Tensor
    ) -> tf.Tensor:
        """
        Normalize audio signal to [-1, 1] range
        Args:
            signal: tf.Tensor with shape[None]
        
        Returns:
            normalized audio signal with shape [None]
        """
        gain = 1.0 / tf.maximum(tf.reduce_max(tf.abs(signal), axis=-1), 1e-9)
        return signal * gain


    def preemphasis(
        self,
        signal: tf.Tensor,
        coeff=0.97
    ):
        """
        Pre-emphasis audio signal
        Args:
            signal: tf.Tensor with shape [None]
            coeff: Float that indicates the preemphasis coefficient
        Returns:
            pre-emphasized signal with shape [None]
        """
        if not coeff or coeff <= 0.0:
            return signal

        s0 = tf.expand_dims(signal[0], axis=-1)
        s1 = signal[1:] - coeff * signal[:-1]
        return tf.concat([s0, s1], axis=-1)


    def get_length_from_duration(self, duration):
        nsamples = math.ceil(float(duration) * self.sample_rate)
        if self.center:
            nsamples += self.nfft
        return 1 + (nsamples - self.nfft) // self.frame_step  # https://www.tensorflow.org/api_docs/python/tf/signal/frame


    def extract_feature(self, signal: tf.Tensor) -> tf.Tensor:
        if self.normalize_signal:
            signal = self.normalize_audio(signal)
        signal = self.preemphasis(signal, self.preemphasis_coeff)

        if self.feature_type == "spectrogram":
            features = self._compute_spectrogram(signal)
        elif self.feature_type == "log_mel_spectrogram":
            features = self._compute_log_mel_spectrogram(signal)
        elif self.feature_type == "mfcc":
            features = self._compute_mfcc(signal)

        features = tf.expand_dims(features, axis=-1)
        if self.normalize_feature:
            features = self.normalize_feature_fn(features, per_frame=self.normalize_per_frame)
        
        return features


    def _stft(self, signal):
        if self.center:
            signal = tf.pad(signal, [[self.nfft // 2, self.nfft // 2]], mode="REFLECT")
        
        window = tf.signal.hann_window(self.frame_length, periodic=True)
        left_pad = (self.nfft - self.frame_length) // 2
        right_pad = self.nfft - self.frame_length - left_pad
        window = tf.pad(window, [[left_pad, right_pad]])

        framed_signal = tf.signal.frame(signal, frame_length=self.nfft, frame_step=self.frame_step)
        framed_signal *= window
        return tf.square(tf.abs(tf.signal.rfft(framed_signal, [self.nfft])))


    def _power_to_db(self, S, amin=1e-10):

        def log10(x):
            return tf.math.log(x) / tf.math.log(tf.constant(10.0))

        log_spec = 10.0 * (log10(tf.maximum(S, amin)) - log10(1.0))
        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - self.top_db)
        return log_spec


    def _compute_spectrogram(self, signal):
        S = self._stft(signal)
        spectrogram = self._power_to_db(S)
        return spectrogram[:, :self.num_feature_bins]


    def _compute_log_mel_spectrogram(self, signal):
        spectrogram = self._stft(signal)
        linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_feature_bins,
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=(self.sample_rate / 2),
        )
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
        return self._power_to_db(mel_spectrogram)


    def _compute_mfcc(self, signal):
        log_mel_spectrogram = self._compute_log_mel_spectrogram(signal)
        return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

