from __future__ import absolute_import

import tensorflow as tf

from models.ctc_base import CtcBase
from layers.conformer_encoder import ConformerEncoder

class ConformerCtc(CtcBase):
    def __init__(
        self,
        vocabulary_size: int,
        encoder_subsampling: dict,
        encoder_positional_encoding: str = "sinusoid",
        encoder_model_depth: int = 144,
        encoder_num_blocks: int = 16,
        encoder_head_size: int = 36,
        encoder_num_heads: int = 4,
        encoder_mha_type: str = "relmha",
        encoder_kernel_size: int = 32,
        encoder_depth_multiplier: int = 1,
        encoder_fc_factor: float = 0.5,
        encoder_dropout: float = 0,
        encoder_trainable: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="conformer_ctc",
        **kwargs
    ):
        super(ConformerCtc, self).__init__(
            encoder=ConformerEncoder(
                subsampling=encoder_subsampling,
                positional_encoding=encoder_positional_encoding,
                model_depth=encoder_model_depth,
                num_blocks=encoder_num_blocks,
                head_size=encoder_head_size,
                num_heads=encoder_num_heads,
                mha_type=encoder_mha_type,
                kernel_size=encoder_kernel_size,
                depth_multiplier=encoder_depth_multiplier,
                fc_factor=encoder_fc_factor,
                dropout=encoder_dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                trainable=encoder_trainable,
                name=f"{name}_encoder",
            ),
            vocabuary_size=vocabulary_size,
            name=name,
            **kwargs
        )

        self.model_depth = encoder_model_depth
        self.time_reduction_factor = self.encoder.time_reduction_factor