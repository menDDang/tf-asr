from __future__ import absolute_import

from tfasr.layers.efficient_net import *
from tfasr.models.transducer_base import TransducerDecoder, TransducerBase
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



class EfficientTransducer(TransducerBase):
    def __init__(self, 
                 vocab_size: int,
                 decoder_attention_dim: int,
                 decoder_num_units: int,
                 model_scale: str = 'b0', 
                 decoder_attention_type: str = "scaled_dot",
                 **kwargs):
        self.vocab_size = vocab_size
        self.decoder_attention_dim = decoder_attention_dim
        self.decoder_num_units = decoder_num_units
        self.model_scale = model_scale
        self.decoder_attention_type = decoder_attention_type
        assert self.model_scale in ['b0', 'b1', 'b2']

        if self.model_scale == 'b0':
            encoder = EfficientNet(
                width_coefficient=1.0,
                depth_coefficient=1.0,
                dropout_rate=0.2,
                blocks_args=BLOCK_ARGS,
                include_top=False)
        elif self.model_scale == 'b1':            
            encoder = EfficientNet(
                width_coefficient=1.0,
                depth_coefficient=1.1,
                dropout_rate=0.2,
                blocks_args=BLOCK_ARGS,
                include_top=False)
        elif self.model_scale == 'b2':         
            encoder = EfficientNet(
                width_coefficient=1.0,
                depth_coefficient=1.2,
                dropout_rate=0.2,
                blocks_args=BLOCK_ARGS,
                include_top=False)

                
        decoder = TransducerDecoder(
            attention_type=self.decoder_attention_type,
            attention_dim=self.decoder_attention_dim,
            num_units=self.decoder_num_units,
            vocab_size=self.vocab_size)

        super(EfficientTransducer, self).__init__(
            encoder=encoder,
            decoder=decoder,
            **kwargs)
        self.time_reduction_factor = 8

    def call(self, inputs, training=False, **kwargs):
        # Parse inputs
        features = inputs["inputs"] # [B, T, D]
        feature_length = inputs["input_lengths"]

        # Encoder outputs
        features = tf.expand_dims(features, axis=2) # [B, T, 1, D]
        encoder_outputs = self.encoder(features, training=training, **kwargs)
        encoder_outputs = tf.squeeze(encoder_outputs, axis=2)

        # Length of encoder outputs
        encoder_output_lengths = tf.cast(
            tf.math.ceil(feature_length / tf.cast(self.time_reduction_factor, dtype=feature_length.dtype)),
            dtype=tf.int32
        )

        
        decoder_inputs = utils.create_inputs(
            inputs=encoder_outputs,
            input_lengths=encoder_output_lengths,
            predictions=inputs["predictions"],
            prediction_lengths=inputs["prediction_lengths"])
        decoder_outputs = self.decoder(decoder_inputs, **kwargs)

        logits = utils.create_logits(
            logits=decoder_outputs,
            logit_lengths=inputs["prediction_lengths"])
        return logits

    def get_config(self):
        super_config =  super().get_config()
        config = {
            "vocab_size": self.vocab_size,
            "model_scale": self.model_scale,
            "decoder_num_units": self.decoder_num_units,
            "decoder_attention_dim": self.decoder_attention_dim,
            "decoder_attention_type": self.decoder_attention_type,
        }
        return {**super_config, **config}