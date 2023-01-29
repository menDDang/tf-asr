from __future__ import absolute_import
from typing import List

import tensorflow as tf


class Expansion(tf.keras.layers.Layer):
    """ MobileNet v2 Expansion phase """
    def __init__(self,
                 expand_ratio: float,
                 activation = 'relu',
                 use_bias: bool = False,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 **kwargs):
        super(Expansion, self).__init__(**kwargs)
        assert expand_ratio >= 1

        self.expand_ratio = expand_ratio
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.bn = tf.keras.layers.BatchNormalization(
            name=f"{self.name}_bn")
        self.act_fn = tf.keras.layers.Activation(
            self.activation,
            name=f"{self.name}_activation")

    def build(self, input_shape):
        super(Expansion, self).build(input_shape)

        assert len(input_shape) == 4

        _, _, _, input_filters = input_shape
        filters = int(input_filters * self.expand_ratio)

        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            data_format="channels_last",
            dilation_rate=(1, 1),
            groups=1,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=f"{self.name}_conv")

    def call(self, x, **kwargs):
        x = self.conv(x, **kwargs)
        x = self.bn(x, **kwargs)
        x = self.act_fn(x, **kwargs)
        return x

    def get_config(self):
        super_config = super(Expansion, self).get_config()
        config = {
            "expand_ratio" : self.expand_ratio,
            "activation" : self.activation,
            "use_bias" : self.use_bias,
            "kernel_initializer" : self.kernel_initializer,
            "bias_initializer" : self.bias_initializer,
            "kernel_regularizer" : self.kernel_regularizer,
            "bias_regularizer" : self.bias_regularizer,
            "kernel_constraint" : self.kernel_constraint,
            "bias_constraint" : self.bias_constraint
        }
        return {**super_config, **config}


class SqueezeExcitation(tf.keras.layers.Layer):
    """ MobileNet v2 Squeeze and Excitation phase """
    def __init__(self,                 
                 input_filters: int, ## block_args.inut_filters
                 se_ratio: float,
                 use_bias: bool = True,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs)
        assert 0 < se_ratio <= 1

        self.input_filters = input_filters
        self.se_ratio = se_ratio
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(
            data_format="channels_last",  # (B, H, W, C)
            name=f"{self.name}_squeeze")
        
        num_reduced_filters = max(1, int(self.input_filters * self.se_ratio))
        
        self.reduce = tf.keras.layers.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            data_format="channels_last",
            dilation_rate=(1, 1),
            groups=1,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=f"{self.name}_reduce")
        


    def build(self, input_shapes):
        super(SqueezeExcitation, self).build(input_shapes)        
        assert len(input_shapes) == 4
        
        batch_size, _, _, filters = input_shapes
        self.batch_size = batch_size
        self.filters = filters
        self.expand = tf.keras.layers.Conv2D(
            self.filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            data_format="channels_last",
            dilation_rate=(1, 1),
            groups=1,
            activation='sigmoid',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=f"{self.name}_expand")

    def call(self, x, **kwargs):
        """
        Arguments: 
            x: input tensor shape of [B, H, W, C]
        """
        # x: [B, C]
        se_tensor = self.avg_pool(x, **kwargs)

        # reduce, shape : [B, 1, 1, C]
        se_tensor = tf.reshape(
            se_tensor, 
            shape=[self.batch_size, 1, 1, self.filters])
        se_tensor = self.reduce(se_tensor)

        # expand, shape : [B, 1, 1, C]
        se_tensor = self.expand(se_tensor)
        x = tf.multiply(x, se_tensor)
        return x

    def get_config(self):
        super_config = super(SqueezeExcitation, self).get_config()
        config = {
            "input_filters" : self.input_filters,
            "se_ratio" : self.se_ratio,
            "use_bias" : self.use_bias,
            "kernel_initializer" : self.kernel_initializer,
            "bias_initializer" : self.bias_initializer,
            "kernel_regularizer" : self.kernel_regularizer,
            "bias_regularizer" : self.bias_regularizer,
            "kernel_constraint" : self.kernel_constraint,
            "bias_constraint" : self.bias_constraint
        }
        return {**super_config, **config}


class MobileInvertedResidualBottleneck(tf.keras.layers.Layer):

    def __init__(self, 
                 output_filters: int,
                 kernel_size: List,
                 strides: List,
                 use_skip_connection: bool,
                 expand_ratio: float,
                 se_ratio: float = None,
                 activation='relu',
                 use_bias: bool = True,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 **kwargs):
        super(MobileInvertedResidualBottleneck, self).__init__(**kwargs)

        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_skip_connection = use_skip_connection
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # Expansion phase
        self.has_expand = self.expand_ratio > 1
        if self.has_expand:
            self.expand = Expansion(
                expand_ratio=self.expand_ratio,
                activation=self.activation,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                name=f"{self.name}_expand")

        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            depth_multiplier=1,
            data_format="channels_last",
            use_bias=False,
            dilation_rate=(1, 1),
            depthwise_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            depthwise_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=f"{self.name}_dwconv")
        self.bn = tf.keras.layers.BatchNormalization(
            name=f"{self.name}_bn")
        self.activation = tf.keras.layers.Activation(
            self.activation,
            name=f"{self.name}_activation")

        # Squeeze and Excitation phase
        self.has_se = (self.se_ratio is not None) and (0 < self.se_ratio <= 1)
        

        # Output phase
        self.proj_conv = tf.keras.layers.Conv2D(
            self.output_filters,
            kernel_size=[1, 1],
            use_bias=False,                
            strides=[1, 1],
            padding='same',
            data_format="channels_last",
            dilation_rate=(1, 1),
            groups=1,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=f"{self.name}_project_conv")
        self.proj_bn = tf.keras.layers.BatchNormalization(
            name=f"{self.name}_project_bn")

        ###### Here, add dropout layer

    def build(self, input_shapes):
        super(MobileInvertedResidualBottleneck, self).build(input_shapes)
        assert(len(input_shapes) == 4)

        _, _, _, input_filters = input_shapes
        self.use_skip_connection &= all(s == 1 for s in self.strides)
        self.use_skip_connection &= (input_filters == self.output_filters)
        
        if self.has_se:
            self.se = SqueezeExcitation(
                input_filters=input_filters,
                se_ratio=self.se_ratio,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                name=f"{self.name}_se")


    def call(self, inputs, **kwargs):
        if self.has_expand:
            x = self.expand(inputs, **kwargs)
        else:
            x = inputs
        
        x = self.dw_conv(x, **kwargs)
        x = self.bn(x, **kwargs)
        x = self.activation(x)

        if self.has_se:
            x = self.se(x, **kwargs)

        x = self.proj_conv(x, **kwargs)
        x = self.proj_bn(x, **kwargs)
        
        # Apply skip connection (residual)
        if self.use_skip_connection:
            x = tf.add(x, inputs)
        return x
            

    def get_config(self):
        super_config = super(MobileInvertedResidualBottleneck, self).get_config()
        config = {
            "kernel_size" : self.kernel_size,
            "strides" : self.strides,
            "expand_ratio" : self.expand_ratio,
            "se_ratio" : self.se_ratio,
            "activation" : self.activation,
            "use_bias" : self.use_bias,
            "kernel_initializer" : self.kernel_initializer,
            "bias_initializer" : self.bias_initializer,
            "kernel_regularizer" : self.kernel_regularizer,
            "bias_regularizer" : self.bias_regularizer,
            "kernel_constraint" : self.kernel_constraint,
            "bias_constraint" : self.bias_constraint
        }
        return {**super_config, **config}



class MobileNetConvBlock(tf.keras.layers.Layer):
    """ Mobile Net Inverted Residual Bottleneck convolution block """
    def __init__(self,
                 output_filters: int,
                 kernel_size: List[int],
                 strides: List[int],
                 expand_ratio: float,
                 se_ratio: float,
                 use_skip_connection: bool,
                 drop_rate: float = None,
                 activation='relu',
                 use_bias: bool = True,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 **kwargs) -> None:
        super(MobileNetConvBlock, self).__init__(**kwargs)

        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.use_skip_connection = use_skip_connection
        self.drop_rate = drop_rate

        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.has_expand = self.expand_ratio > 1
        self.has_se = (self.se_ratio is not None) and (0 < self.se_ratio <= 1)


    def build(self, input_shape: List) -> None:
        assert len(input_shape) == 4
        _, _, _, input_filters = input_shape
        
        self.use_skip_connection &= all(s == 1 for s in self.strides)
        self.use_skip_connection &= (input_filters == self.output_filters)

        # Expansion phase
        filters = int(input_filters * self.expand_ratio)
        if self.has_expand:
            self.expand_conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="same",
                data_format="channels_last",
                dilation_rate=[1, 1],
                groups=1,
                activation=None,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                name=f"{self.name}_expand_conv")            
            self.expand_bn = tf.keras.layers.BatchNormalization(
                name=f"{self.name}_expand_bn")
            self.expand_activation = tf.keras.layers.Activation(
                self.activation,
                name=f"{self.name}_expand_activation")

        # Depth-wise convolution
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            depth_multiplier=1,
            data_format="channels_last",
            use_bias=False,
            dilation_rate=(1, 1),
            depthwise_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            depthwise_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=f"{self.name}_dwconv")
        self.bn = tf.keras.layers.BatchNormalization(
            name=f"{self.name}_bn")
        self.activation = tf.keras.layers.Activation(
            self.activation,
            name=f"{self.name}_activation")

        # 
        if self.has_se:
            self.se_squeeze = tf.keras.layers.GlobalAveragePooling2D(
                data_format="channels_last",  # (B, H, W, C)
                name=f"{self.name}_se_squeeze")
            num_reduced_filters = max(1, int(input_filters * self.se_ratio))
            self.se_reduce = tf.keras.layers.Conv2D(
                num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same',
                data_format="channels_last",
                dilation_rate=(1, 1),
                groups=1,
                activation=None,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                name=f"{self.name}_se_reduce")
            self.se_expand = tf.keras.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same',
                data_format="channels_last",
                dilation_rate=(1, 1),
                groups=1,
                activation='sigmoid',
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                name=f"{self.name}_expand")
        
        # Output phase
        self.proj_conv = tf.keras.layers.Conv2D(
            self.output_filters,
            kernel_size=[1, 1],
            use_bias=False,                
            strides=[1, 1],
            padding='same',
            data_format="channels_last",
            dilation_rate=(1, 1),
            groups=1,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=f"{self.name}_project_conv")
        self.proj_bn = tf.keras.layers.BatchNormalization(
            name=f"{self.name}_project_bn")

        if self.use_skip_connection:
            if self.drop_rate is not None and self.drop_rate > 0:
                self.dropout = tf.keras.layers.Dropout(
                    self.drop_rate,
                    noise_shape=[None, 1, 1, 1],
                    name=f"{self.name}_dropout")

        super(MobileNetConvBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.has_expand:
            x = self.expand_conv(inputs, **kwargs)
            x = self.expand_bn(x, **kwargs)
            x = self.expand_activation(x, **kwargs)
        else:
            x = inputs

        x = self.dw_conv(x, **kwargs)
        x = self.bn(x, **kwargs)
        x = self.activation(x)

        if self.has_se:
            se_inputs = x
            x = self.se_squeeze(x, **kwargs) # [B, C]
            _shape = tf.shape(x)
            B = tf.gather(_shape, 0)
            C = tf.gather(_shape, 1)
            x = tf.reshape(x, [B, 1, 1, C])
            x = self.se_reduce(x, **kwargs)
            x = self.se_expand(x, **kwargs)
            x = tf.multiply(x, se_inputs)

        x = self.proj_conv(x, **kwargs)
        x = self.proj_bn(x, **kwargs)

        if self.use_skip_connection:
            if self.drop_rate is not None and self.drop_rate > 0:
                x = self.dropout(x, **kwargs)
            x = tf.add(x, inputs)

        return x


    def get_config(self):
        super_config = super(MobileNetConvBlock, self).get_config()
        config = {
            "output_filters": self.output_filters,
            "kernel_size" : self.kernel_size,
            "strides" : self.strides,
            "expand_ratio" : self.expand_ratio,
            "se_ratio" : self.se_ratio,
            "use_skip_connection": self.use_skip_connection,
            "drop_rate": self.drop_rate,
            "activation" : self.activation,
            "use_bias" : self.use_bias,
            "kernel_initializer" : self.kernel_initializer,
            "bias_initializer" : self.bias_initializer,
            "kernel_regularizer" : self.kernel_regularizer,
            "bias_regularizer" : self.bias_regularizer,
            "kernel_constraint" : self.kernel_constraint,
            "bias_constraint" : self.bias_constraint
        }
        return {**super_config, **config}