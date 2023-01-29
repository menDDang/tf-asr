from __future__ import absolute_import

import tensorflow as tf

from layers.gated_linear_units import GLU
from layers.positional_encoding import PositionalEncoding
from layers.multi_head_attention import MultiHeadAttention, RelPositionMultiHeadAttention
from layers.subsampling import Conv2dSubsampling


class FFModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        dropout=0.0,
        fc_factor=0.5,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="ff_module",
        **kwargs
    ):
        super(FFModule, self).__init__(name=name, **kwargs)

        self.fc_factor = fc_factor
        self.layer_norm = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        self.ffn1 = tf.keras.layers.Dense(
            input_dim * 4,
            activation=tf.nn.swish,
            name=f"{name}_dense_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.dropout1 = tf.keras.layers.Dropout(
            dropout,
            name=f"{name}_dropout_1"
        )
        self.ffn2 = tf.keras.layers.Dense(
            input_dim,
            name=f"{name}_dense_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.dropout2 = tf.keras.layers.Dropout(
            dropout,
            name=f"{name}_dropout_2"
        )
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(
        self,
        inputs,
        training=False,
        **kwargs
    ):
        outputs = self.layer_norm(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.dropout1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.dropout2(outputs, training=training)
        outputs = self.res_add([inputs, self.fc_factor * outputs])
        return outputs

    def get_config(self):
        conf = super(FFModule, self).get_config()
        conf.update({"fc_factor": self.fc_factor})
        conf.update(self.ln.get_config())
        conf.update(self.ffn1.get_config())
        conf.update(self.dropout1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.dropout2.get_config())
        conf.update(self.res_add.get_config())
        return conf


class MHAModule(tf.keras.layers.Layer):
    def __init__(
        self,
        head_size,
        num_heads,
        dropout=0.0,
        mha_type="relmha",
        kernel_regularizer=None,
        bias_regularizer=None,
        name="mhsa_module",
        **kwargs
    ):
        super(MHAModule, self).__init__(name=name, **kwargs)

        self.layer_norm = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
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
        self.dropout = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")
        self.mha_type = mha_type

    def call(
        self,
        inputs,
        training=False,
        mask=None,
        **kwargs,
    ):
        inputs, positional_eoncodings = inputs
        outputs = self.layer_norm(inputs, training=training)
        if self.mha_type == "relmha":
            outputs = self.mha([outputs, outputs, outputs, positional_eoncodings], training=training, mask=mask)
        else:
            outputs = outputs + positional_eoncodings
            outputs = self.mha([outputs, outputs, outputs], training=training, mask=mask)
        outputs = self.dropout(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(MHAModule, self).get_config()
        conf.update({"mha_type": self.mha_type})
        conf.update(self.layer_norm.get_config())
        conf.update(self.mha.get_config())
        conf.update(self.dropout.get_config())
        conf.update(self.res_add.get_config())
        return conf


class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        kernel_size=32,
        dropout=0.0,
        depth_multiplier=1,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="conv_module",
        **kwargs
    ):
        super(ConvModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        self.pw_conv_1 = tf.keras.layers.Conv2D(
            filters=input_dim * 2,
            kernel_size=1,
            strides=1,
            padding="valid",
            name=f"{name}_pw_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.glu = GLU(name=f"{name}_glu")
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, 1),
            strides=1,
            padding="same",
            name=f"{name}_dw_conv",
            depth_multiplier=depth_multiplier,
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name=f"{name}_bn",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        self.swish = tf.keras.layers.Activation(
            tf.nn.swish,
            name=f"{name}_swish_activation"
        )
        self.pw_conv_2 = tf.keras.layers.Conv2D(
            filters=input_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name=f"{name}_pw_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(
        self,
        inputs,
        training=False,
        **kwargs
    ):
        outputs = self.ln(inputs, training=training)
        #B, T, E = inputs.shape
        #outputs = tf.reshape(outputs, [B, T, 1, E])
        outputs = tf.expand_dims(outputs, axis=2)
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        #outputs = tf.reshape(outputs, [B, T, E])
        outputs = tf.squeeze(outputs, axis=2)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(ConvModule, self).get_config()
        conf.update(self.ln.get_config())
        conf.update(self.pw_conv_1.get_config())
        conf.update(self.glu.get_config())
        conf.update(self.dw_conv.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.pw_conv_2.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        dropout=0.0,
        fc_factor=0.5,
        head_size=36,
        num_heads=4,
        mha_type="relmha",
        kernel_size=32,
        depth_multiplier=1,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="conformer_block",
        **kwargs,
    ):
        super(ConformerBlock, self).__init__(name=name, **kwargs)
        self.ffm1 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            name=f"{name}_ff_module_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.mhsam = MHAModule(
            mha_type=mha_type,
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            name=f"{name}_mha_module",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.convm = ConvModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            name=f"{name}_conv_module",
            depth_multiplier=depth_multiplier,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.ffm2 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
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

class ConformerEncoder(tf.keras.Model):
    def __init__(
        self,
        subsampling: dict,
        positional_encoding="sinsuoid",
        model_depth=144,
        num_blocks=16,
        mha_type="relmha",
        head_size=36,
        num_heads=4,
        kernel_size=32,
        depth_multiplier=1,
        fc_factor=0.5,
        dropout=0.0,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="conformer_encoder",
        **kwargs
    ):
        super(ConformerEncoder, self).__init__(name=name, **kwargs)

        # Sub-sampling layer
        subsampling_name = subsampling.pop("type", "conv2d")
        if subsampling_name == "vgg":
            raise ValueError("Not implemented yet.")
        elif subsampling_name == "conv2d":
            subsampling_class = Conv2dSubsampling
        else:
            raise ValueError("subsampling must be either  'conv2d' or 'vgg'")

        self.conv_subsampling = subsampling_class(
            **subsampling,
            name=f"{name}_subsampling",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.time_reduction_factor = self.conv_subsampling.time_reduction_factor

        # Positional Encoding layer
        if positional_encoding == "sinsuoid":
            self.pe = PositionalEncoding(mode="add", name=f"{name}_pe")
        elif positional_encoding == "sinsuoid_v2":
            self.pe = PositionalEncoding(mode="add", alpha=2, beta=0, name=f"{name}_pe")
        elif positional_encoding == "sinusoid_concat":
            self.pe = PositionalEncoding(mode="concat", name=f"{name}_pe")
        elif positional_encoding == "sinusoid_concat_v2":
            self.pe = PositionalEncoding(mode="concat", alpha=2, beta=-1, name=f"{name}_pe")
        elif positional_encoding == "subsampling":
            self.pe = tf.keras.layers.Activation("linear", name=f"{name}_pe")
        else:
            raise ValueError(
                "positional_encoding must be either 'sinusoid', 'sinusoid_concat', 'sinusoid_v2', 'sinusoid_concat_v2' or 'subsampling'"
            )
 
        self.linear = tf.keras.layers.Dense(
            model_depth,
            name=f"{name}_linear",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

        self.conformer_blocks = [
            ConformerBlock(
                input_dim=model_depth,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                mha_type=mha_type,
                kernel_size=kernel_size,
                depth_multiplier=depth_multiplier,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_block_{i}"
            )
            for i in range(num_blocks)
        ]

    def call(
        self,
        inputs,
        training=False,
        mask=None,
        **kwargs
    ):
        # input with shape [B, T, D, 1]
        outputs = self.conv_subsampling(inputs, training=training)
        outputs = self.linear(outputs, training=training)
        pe = self.pe(outputs)
        outputs = self.do(outputs, training=training)
        for block in self.conformer_blocks:
            outputs = block([outputs, pe], training=training, mask=mask, **kwargs)
        return outputs
    
    def get_config(self):
        conf = super(ConformerEncoder, self).get_config()
        conf.update(self.conv_subsampling.get_config())
        conf.update(self.linear.get_config())
        conf.update(self.do.get_config())
        conf.update(self.pe.get_config())
        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf