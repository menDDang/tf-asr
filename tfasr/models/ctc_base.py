from __future__ import absolute_import
from typing import Union, Dict

import tensorflow as tf

from tfasr.models.model_base import ModelBase
from tfasr import utils


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, blank=0, global_batch_size=None, name=None):
        super(CTCLoss, self).__init__(
            reduction=tf.keras.losses.Reduction.NONE,
            name=name
        )

        self.blank = blank
        self.global_batch_size = global_batch_size

    @tf.function
    def call(self, y_true, y_pred):
        label_length = y_true["label_lengths"]
        labels = y_true["labels"]

        logit_length = y_pred["logit_lengths"]
        logits = y_pred["logits"]

        loss = tf.nn.ctc_loss(
            label_length=tf.cast(label_length, tf.int32),
            labels=tf.cast(labels, tf.int32),
            logit_length=tf.cast(logit_length, tf.int32),
            logits=tf.cast(logits, tf.float32),
            logits_time_major=False,
            blank_index=self.blank
        )
        return tf.nn.compute_average_loss(
            loss,
            global_batch_size=self.global_batch_size
        )


class CTCBase(ModelBase):

    def __init__(self,
                 encoder: tf.keras.Model,
                 decoder: Union[tf.keras.Model, tf.keras.layers.Layer] = None,
                 vocabuary_size: int = None,
                 **kwargs):
        super(CTCBase, self).__init__(**kwargs)

        self.encoder = encoder
        if decoder is None:
            assert vocabuary_size is not None, "vocabulary_size must be set."

            self.decoder = tf.keras.layers.Dense(
                units=vocabuary_size,
                name=f"{self.name}_logits"
            )
        else:
            self.decoder = decoder
        self.time_reduction_factor = 1

    def compile(self, 
                optimizer, 
                global_batch_size, 
                blank=0, 
                run_eagerly=None, 
                **kwargs):
        loss_fn = CTCLoss(blank=blank, global_batch_size=global_batch_size)
        return super().compile(loss_fn, optimizer, run_eagerly=run_eagerly, **kwargs)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        encoder_outputs = self.encoder(inputs["inputs"], training=training, **kwargs)
        logits = self.decoder(encoder_outputs, training=training, **kwargs)

        length = inputs["input_lengths"]
        reduced_length = tf.cast(
            tf.math.ceil(length / tf.cast(self.time_reduction_factor, dtype=length.dtype)),
            dtype=tf.int32
        )

        return utils.create_logits(logits=logits, logit_lengths=reduced_length)
    
        
    def infer_greedy(self, inputs: Dict[str, tf.Tensor]):
        """Greedy decoding function that used in self.predict_step"""
        logits = self.call(inputs, training=False)
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=logits["logits"],
            sequence_length=logits["logit_lengths"],
            merge_repeated=True,
            blank_index=0
        )
        return decoded[0]