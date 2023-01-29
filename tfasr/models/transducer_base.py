from __future__ import absolute_import

from typing import Union, Dict

import tensorflow as tf

from tfasr.models.model_base import ModelBase
from tfasr import utils

       

class CharacterDistrubutionLoss(tf.keras.losses.Loss):
    def __init__(self, global_batch_size=None, name=None):
        super(CharacterDistrubutionLoss, self).__init__(
            reduction=tf.keras.losses.Reduction.NONE,
            name=name
        )

        self.global_batch_size = global_batch_size

    def call(self, y_true, y_pred):
        label_length = y_true["label_lengths"]
        labels = y_true["labels"]

        logit_length = y_pred["logit_lengths"]
        logits = y_pred["logits"]

        # loss : [B, U]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

        mask = tf.sequence_mask(
            label_length,
            maxlen=tf.gather(tf.shape(loss), 1))
        loss = tf.where(mask, loss, tf.zeros_like(loss))

        # loss : [B]
        loss = tf.reduce_sum(loss, axis=1)
        #loss /= tf.cast(label_length, tf.float32)

        return tf.nn.compute_average_loss(
            loss,
            global_batch_size=self.global_batch_size
        )


class TransducerDecoder(tf.keras.layers.Layer):
    def __init__(self, 
                 attention_type: str,
                 attention_dim: int,
                 num_units: int,
                 vocab_size: int):
        super(TransducerDecoder, self).__init__()
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.num_units = num_units
        self.vocab_size = vocab_size

        self.proj_query = tf.keras.layers.Dense(self.attention_dim)
        self.proj_key = tf.keras.layers.Dense(self.attention_dim)

        # Set attention
        if self.attention_type == "dot":
            self.attention = tf.keras.layers.Attention(use_scale=False)
        elif self.attention_type == "scaled_dot":
            self.attention = tf.keras.layers.Attention(use_scale=True)
        else:
            raise ValueError("Invalid attention type")
        
        # Set RNN
        self.rnn1 = tf.keras.layers.GRU(self.num_units, return_sequences=True)        
        self.dense = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, **kwargs):
        x = inputs["inputs"]
        predictions = inputs["predictions"]
        predictions = tf.one_hot(
            predictions, 
            depth=self.vocab_size,
            dtype=tf.float32)
        query = self.proj_query(predictions, **kwargs)
        key = self.proj_key(x, **kwargs)

        query_mask = tf.sequence_mask(
            inputs["prediction_lengths"],
            maxlen=tf.gather(tf.shape(query), 1))
        value_mask = tf.sequence_mask(
            inputs["input_lengths"],
            maxlen=tf.gather(tf.shape(key), 1))
        attention_context = self.attention(
            inputs=[query, key], 
            mask=[query_mask, value_mask],
            **kwargs)
                
        out = self.rnn1(attention_context, **kwargs)
        out = self.dense(out, **kwargs)
        return out
        
class TransducerBase(ModelBase):

    def __init__(self,
                 encoder: Union[tf.keras.layers.Layer, tf.keras.Model],
                 decoder: Union[tf.keras.layers.Layer, tf.keras.Model],
                 **kwargs):
        super(TransducerBase, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.time_reduction_factor = 1

    def compile(self, 
                optimizer, 
                global_batch_size, 
                **kwargs):
        loss_fn = CharacterDistrubutionLoss(global_batch_size=global_batch_size)
        return super().compile(loss_fn, optimizer, **kwargs)

    def call(self, inputs, **kwargs):
        feature = inputs["inputs"]
        feature_length = inputs["input_lengths"]

        encoder_outputs = self.encoder(feature, **kwargs)
        encoder_output_lengths = tf.cast(
            tf.math.ceil(feature_length / self.time_reduction_factor),
            dtype=tf.int32)

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
        super_config = super(TransducerBase, self).get_config()
        config = {
            "encoder" : self.encoder,
            "decoder" : self.decoder
        }
        return {**super_config, **config}
