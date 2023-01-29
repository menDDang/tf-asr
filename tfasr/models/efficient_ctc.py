from __future__ import absolute_import

from tfasr.models.ctc_base import CTCBase
from tfasr.layers.efficient_net import *
from tfasr import utils


BLOCK_ARGS = [
    BlockArgs(kernel_size=[3, 1], num_repeat=1, output_filters=16,
        expand_ratio=1, use_skip_connection=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=[3, 1], num_repeat=2, output_filters=24,
        expand_ratio=6, use_skip_connection=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=[5, 1], num_repeat=2, output_filters=40,
        expand_ratio=6, use_skip_connection=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=[3, 1], num_repeat=3, output_filters=80,
        expand_ratio=6, use_skip_connection=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=[5, 1], num_repeat=3, output_filters=112,
        expand_ratio=6, use_skip_connection=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=[5, 1], num_repeat=4,  output_filters=192,
        expand_ratio=6, use_skip_connection=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=[3, 1], num_repeat=1,  output_filters=320,
        expand_ratio=6, use_skip_connection=True, strides=[1, 1], se_ratio=0.25)
]


class EfficientCTC(CTCBase):
    def __init__(self, 
                 vocab_size: int,
                 model_scale: str = 'b0', 
                 num_gru_units: int = 512,
                 **kwargs):
        self.vocab_size = vocab_size
        self.model_scale = model_scale
        self.num_gru_units = num_gru_units

        if self.model_scale == 'b0':
            encoder = EfficientNet(
                width_coefficient=1.0,
                depth_coefficient=1.0,
                dropout_rate=0.2,
                blocks_args=BLOCK_ARGS,
                include_top=False)
                
        decoder = tf.keras.Sequential()
        decoder.add(tf.keras.layers.GRU(self.num_gru_units, return_sequences=True))
        decoder.add(tf.keras.layers.Dense(self.vocab_size))

        super(EfficientCTC, self).__init__(
            encoder=encoder,
            decoder=decoder,
            **kwargs)
        self.time_reduction_factor = 8

    def call(self, inputs, training=False, **kwargs):
        features = inputs["inputs"] # [B, T, D]
        features = tf.expand_dims(features, axis=2) # [B, T, 1, D]

        encoder_outputs = self.encoder(features, training=training, **kwargs)
        shape = tf.shape(encoder_outputs)
        B = tf.gather(shape, 0)
        T = tf.gather(shape, 1)
        encoder_outputs = tf.reshape(encoder_outputs, [B, T, -1])
        logits = self.decoder(encoder_outputs, training=training, **kwargs)

        length = inputs["input_lengths"]
        reduced_length = tf.cast(
            tf.math.ceil(length / tf.cast(self.time_reduction_factor, dtype=length.dtype)),
            dtype=tf.int32
        )

        return utils.create_logits(logits=logits, logit_lengths=reduced_length)
    