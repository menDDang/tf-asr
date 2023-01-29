from __future__ import absolute_import

import tensorflow as tf

from layers.gated_linear_units import GLU
from layers.multi_head_attention import MultiHeadAttention, RelPositionMultiHeadAttention

class FeedForwardModule(tf.keras.layers.Layer):
    def __init__(
        self, 
        input_dim, 
        dropout_prob=0.0,
        fc_factor=0.5,
        kernel_regularizer=tf.keras.regularizers.l2(1e-6),
        bias_regularizer=tf.keras.regularizers.l2(1e-6),
        name="ff_module",
        **kwargs
    ):

        super(FeedForwardModule, self).__init__(name=name, **kwargs)

        self.fc_factor = fc_factor
        self.layer_norm = tf.keras.layers.LayerNormalization(
            name=f"{name}_layer_norm",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        self.ffn1 = tf.keras.layers.Dense(
            4 * input_dim,
            name=f"{name}_ffn1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.swish = tf.keras.layers.Activation(
            tf.nn.swish,
            name=f"{name}_swish"
        )
        self.dropout1 = tf.keras.layers.Dropout(
            dropout_prob,
            name=f"{name}_dropout_1"
        )
        self.ffn2 = tf.keras.layers.Dense(
            input_dim,
            name=f"{name}_dense_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.dropout2 = tf.keras.layers.Dropout(
            dropout_prob,
            name=f"{name}_dropout_2"
        )
        self.residual_add = tf.keras.layers.Add(
            name=f"{name}_add"
        )

    def call(self, inputs, training=False, **kwargs):
        outputs = self.layer_norm(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.dropout1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.dropout2(outputs, training=training)
        outputs = self.residual_add([inputs, self.fc_factor * outputs])
        return outputs



class MultiHeadSelfAttentionModule(tf.keras.layers.Layer):
    def __init__(
        self,
        head_size,
        num_heads,
        dropout_prob=0.0,
        mha_type="relmha",
        kernel_regularizer=tf.keras.regularizers.l2(1e-6),
        bias_regularizer=tf.keras.regularizers.l2(1e-6),
        name="mhsa_module",
        **kwargs
    ):
        super(MultiHeadSelfAttentionModule, self).__init__(name=name, **kwargs)

        self.layer_norm = tf.keras.layers.LayerNormalization(
            name=f"{name}_layer_norm",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )

        if mha_type == "relmha":
            self.mha = RelPositionMultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size,
                num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        elif mha_type == "mha":
            self.mha = MultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size,
                num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        else:
            raise ValueError("mha_type must be either 'mha' or 'relmha'")

        self.dropout = tf.keras.layers.Dropout(
            dropout_prob,
            name=f"{name}_dropout"
        )
        self.residual_add = tf.keras.layers.Add(name=f"{name}_add")
        self.mha_type = mha_type

    def call(
        self,
        inputs,
        training=False,
        mask=None,
        **kwargs
    ):
        # inputs : [inputs, positional_encodigs]
        inputs, pos = inputs
        outputs = self.layer_norm(inputs, training=training)
        if self.mha_type == "relmha":
            outputs = self.mha(
                [outputs, outputs, outputs, pos],
                training=training,
                mask=mask
            )
        else:
            outputs = outputs + pos
            outputs = self.mha(
                [outputs, outputs, outputs],
                training=training,
                mask=mask
            )
        outputs = self.dropout(outputs, training=training)
        outputs = self.residual_add([inputs, outputs])
        return outputs


class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        kernel_size=32,
        dropout_prob=0.0,
        depth_multiplier=1,
        kernel_regularizer=tf.keras.regularizers.l2(1e-6),
        bias_regularizer=tf.keras.regularizers.l2(1e-6),
        name="conv_module",
        **kwargs
    ):
        super(ConvModule, self).__init__(name=name, **kwargs)

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.pointwise_conv_1 = tf.keras.layers.Conv2D(
            filters=2*input_dim,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="valid",
            name=f"{name}_pointwise_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.glu = GLU(name=f"{name}_glu")
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=[kernel_size, 1],
            strides=[1, 1],
            padding="same",
            name=f"{name}_depthwise_conv",
            depth_multiplier=depth_multiplier,
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.batch_norm = tf.keras.layers.BatchNormalization(
            name=f"{name}_batch_norm",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.swish = tf.keras.layers.Activation(
            tf.nn.swish,
            name=f"{name}_swish_activation"
        )
        self.pointwise_conv_2 = tf.keras.layers.Conv2D(
            filters=input_dim,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="valid",
            name=f"{name}_pintwise_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.dropout = tf.keras.layers.Dropout(
            dropout_prob,
            name=f"{name}_dropout"
        )
        self.residual_add = tf.keras.layers.Add(
            name=f"{name}_add"
        )
        
    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        outputs = self.layer_norm(inputs, training=training)
        # outputs: [B, T, E] --> [B, T, 1, E]
        output_shape = tf.shape(outputs)
        B, T, E = output_shape[0], output_shape[1], output_shape[2]
        outputs = tf.reshape(outputs, [B, T, 1, E])
        outputs = self.pointwise_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.depthwise_conv(outputs, training=training)
        outputs = self.batch_norm(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pointwise_conv_2(outputs, training=training)
        outputs = tf.reshape(outputs, [B, T, E])
        outputs = self.dropout(outputs, training=training)
        outputs = self.residual_add([inputs, outputs])
        return outputs



class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        dropout_prob=0.0,
        fc_factor=0.5,
        head_size=36,
        num_heads=4,
        mha_type="relmha",
        kernel_size=32,
        depth_multiplier=1,
        kernel_regularizer=tf.keras.regularizers.l2(1e-6),
        bias_regularizer=tf.keras.regularizers.l2(1e-6),
        name="conformer_block",
        **kwargs,
    ):
        super(ConformerBlock, self).__init__(name=name, **kwargs)
        self.ffm1 = FeedForwardModule(
            input_dim=input_dim,
            dropout_prob=dropout_prob,
            fc_factor=fc_factor,
            name=f"{name}_ff_module_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.mhsam = MultiHeadSelfAttentionModule(
            mha_type=mha_type,
            head_size=head_size,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            name=f"{name}_mhsa_module",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.convm = ConvModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            dropout_prob=dropout_prob,
            name=f"{name}_conv_module",
            depth_multiplier=depth_multiplier,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.ffm2 = FeedForwardModule(
            input_dim=input_dim,
            dropout_prob=dropout_prob,
            fc_factor=fc_factor,
            name=f"{name}_ff_module_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=kernel_regularizer,
        )

    def call(
        self,
        inputs,
        training=False,
        mask=None,
        **kwargs,
    ):
        inputs, pos = inputs  # pos is positional encoding
        outputs = self.ffm1(inputs, training=training, **kwargs)
        outputs = self.mhsam([outputs, pos], training=training, mask=mask, **kwargs)
        outputs = self.convm(outputs, training=training, **kwargs)
        outputs = self.ffm2(outputs, training=training, **kwargs)
        outputs = self.ln(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(ConformerBlock, self).get_config()
        conf.update(self.ffm1.get_config())
        conf.update(self.mhsam.get_config())
        conf.update(self.convm.get_config())
        conf.update(self.ffm2.get_config())
        conf.update(self.ln.get_config())
        return conf


