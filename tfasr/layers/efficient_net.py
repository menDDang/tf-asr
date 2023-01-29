from __future__ import absolute_import

import math
import collections
from multiprocessing import pool

import tensorflow as tf

from tfasr.layers.mobile_net import MobileNetConvBlock


BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'output_filters',
    'expand_ratio', 'use_skip_connection', 'strides', 'se_ratio'
])

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, output_filters=16,
              expand_ratio=1, use_skip_connection=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, output_filters=24,
              expand_ratio=6, use_skip_connection=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, output_filters=40,
              expand_ratio=6, use_skip_connection=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, output_filters=80,
              expand_ratio=6, use_skip_connection=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, output_filters=112,
              expand_ratio=6, use_skip_connection=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4,  output_filters=192,
              expand_ratio=6, use_skip_connection=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1,  output_filters=320,
              expand_ratio=6, use_skip_connection=True, strides=[1, 1], se_ratio=0.25)
]




class EfficientNet(tf.keras.layers.Layer):
    def __init__(self,
                 width_coefficient,
                 depth_coefficient,                 
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args: BlockArgs = DEFAULT_BLOCKS_ARGS,
                 include_top: bool =True,
                 pooling=None,
                 classes=1000,
                 activation = tf.keras.activations.swish,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 name='efficientnet',
                 **kwargs):
        super(EfficientNet, self).__init__(name=name, **kwargs)
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.blocks_args = blocks_args
        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes
        self.activation = activation
        
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # Build stem
        self.stem_conv = tf.keras.layers.Conv2D(
            self.round_filters(32),
            kernel_size=[3, 3],
            use_bias=False,
            strides=[2, 2],
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
            name=f"{self.name}_stem_conv")
        self.stem_bn = tf.keras.layers.BatchNormalization(
            name=f"{self.name}_stem_bn")
        self.stem_activation = tf.keras.layers.Activation(
            self.activation,
            name=f"{self.name}_stem_activation")
        
        # Build blocks
        self.mobile_conv_blocks = tf.keras.Sequential()
        num_blocks_total = sum(
            self.round_repeats(block_args.num_repeat)
            for block_args in self.blocks_args)

        block_num = 0
        for idx, block_args in enumerate(self.blocks_args):
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier
            block_args = block_args._replace(
                output_filters=self.round_filters(
                    block_args.output_filters),
                num_repeat=self.round_repeats(
                    block_args.num_repeat))

            # The first block needs to take care of stride and filter size increase
            dropout_rate = self.drop_connect_rate * float(block_num) / num_blocks_total
            self.mobile_conv_blocks.add(MobileNetConvBlock(
                output_filters=block_args.output_filters,
                kernel_size=block_args.kernel_size,
                strides=block_args.strides,
                use_skip_connection=block_args.use_skip_connection,
                expand_ratio=block_args.expand_ratio,
                se_ratio=block_args.se_ratio,
                drop_rate=dropout_rate,
                activation=self.activation,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                name=f"{self.name}_block{idx+1}_1"))
            block_num += 1
            
            if block_args.num_repeat > 1:
                for i in range(1, block_args.num_repeat):
                    dropout_rate = self.drop_connect_rate * float(block_num) / num_blocks_total
                    self.mobile_conv_blocks.add(MobileNetConvBlock(
                        output_filters=block_args.output_filters,
                        kernel_size=block_args.kernel_size,
                        strides=[1, 1],
                        use_skip_connection=block_args.use_skip_connection,
                        expand_ratio=block_args.expand_ratio,
                        se_ratio=block_args.se_ratio,
                        drop_rate=dropout_rate,
                        activation=self.activation,
                        use_bias=False,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint,
                        name=f"{self.name}_block{idx+1}_{i+1}"))
                    block_num += 1
            
        # Build top
        self.top = tf.keras.Sequential()
        self.top.add(
        tf.keras.layers.Conv2D(
            self.round_filters(1280),
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
            name=f"{self.name}_top_conv"))
        self.top.add(
            tf.keras.layers.BatchNormalization(
                name=f"{self.name}_top_bn"))
        self.top.add(
            tf.keras.layers.Activation(
                self.activation,
                name=f"{self.name}_top_activation"))
            
        if self.include_top:
            self.top.add(
                tf.keras.layers.GlobalAveragePooling2D(
                    data_format="channels_last",  # (B, H, W, C)
                    keepdims=False,
                    name=f"{self.name}_squeeze"))
            self.top.add(
                tf.keras.layers.Dense(
                    self.classes,
                    activation='softmax',
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    kernel_constraint=self.kernel_constraint))
        else:
            if self.pooling == 'avg':
                self.top.add(tf.keras.layers.GlobalAveragePooling2D(
                    data_format="channels_last",  # (B, H, W, C)
                    keepdims=False,
                    name=f"{self.name}_squeeze"))
            elif pooling == "max":
                self.top.add(tf.keras.layers.GlobalMaxPooling2D(
                    data_format="channels_last",  # (B, H, W, C)
                    keepdims=False,
                    name=f"{self.name}_squeeze"))

    def call(self, x, **kwargs):
        x = self.stem_conv(x, **kwargs)
        x = self.stem_bn(x, **kwargs)
        x = self.stem_activation(x, **kwargs)        
        x = self.mobile_conv_blocks(x, **kwargs)
        x = self.top(x, **kwargs)
        return x

    def round_filters(self, filters):
        """ Round number of filters based on width multiplier. """
        filters *= self.width_coefficient
        new_filters = int(filters + self.depth_divisor / 2) // self.depth_divisor * self.depth_divisor
        new_filters = max(self.depth_divisor, new_filters)
        if new_filters < 0.9 * filters:
            new_filters += self.depth_divisor
        return int(new_filters)

    def round_repeats(self, repeats):
        """ Round number of repeats based on depth multiplier. """
        return int(math.ceil(self.depth_coefficient * repeats))


class EfficientNetB0(EfficientNet):
    def __init__(self, 
                 include_top=True,
                 pooling: str = None,
                 classes: int = 1000,
                 **kwargs):
        width_coeff = 1.0
        depth_coeff = 1.0
        dropout_rate = 0.2
        super(EfficientNetB0, self).__init__(
            width_coefficient=width_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            include_top=include_top,
            pooling=pooling,
            classes=classes,
            **kwargs)


class EfficientNetB1(EfficientNet):
    def __init__(self, 
                 include_top=True,
                 pooling: str = None,
                 classes: int = 1000,
                 **kwargs):
        width_coeff = 1.0
        depth_coeff = 1.1
        dropout_rate = 0.2
        super(EfficientNetB1, self).__init__(
            width_coefficient=width_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            include_top=include_top,
            pooling=pooling,
            classes=classes,
            **kwargs)

class EfficientNetB2(EfficientNet):
    def __init__(self, 
                 include_top=True,
                 pooling: str = None,
                 classes: int = 1000,
                 **kwargs):
        width_coeff = 1.0
        depth_coeff = 1.2
        dropout_rate = 0.3
        super(EfficientNetB2, self).__init__(
            width_coefficient=width_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            include_top=include_top,
            pooling=pooling,
            classes=classes,
            **kwargs)

class EfficientNetB3(EfficientNet):
    def __init__(self, 
                 include_top=True,
                 pooling: str = None,
                 classes: int = 1000,
                 **kwargs):
        width_coeff = 1.2
        depth_coeff = 1.4
        dropout_rate = 0.3
        super(EfficientNetB3, self).__init__(
            width_coefficient=width_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            include_top=include_top,
            pooling=pooling,
            classes=classes,
            **kwargs)

class EfficientNetB4(EfficientNet):
    def __init__(self, 
                 include_top=True,
                 pooling: str = None,
                 classes: int = 1000,
                 **kwargs):
        width_coeff = 1.4
        depth_coeff = 1.8
        dropout_rate = 0.4
        super(EfficientNetB4, self).__init__(
            width_coefficient=width_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            include_top=include_top,
            pooling=pooling,
            classes=classes,
            **kwargs)

class EfficientNetB5(EfficientNet):
    def __init__(self, 
                 include_top=True,
                 pooling: str = None,
                 classes: int = 1000,
                 **kwargs):
        width_coeff = 1.6
        depth_coeff = 2.2
        dropout_rate = 0.4
        super(EfficientNetB5, self).__init__(
            width_coefficient=width_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            include_top=include_top,
            pooling=pooling,
            classes=classes,
            **kwargs)

class EfficientNetB6(EfficientNet):
    def __init__(self, 
                 include_top=True,
                 pooling: str = None,
                 classes: int = 1000,
                 **kwargs):
        width_coeff = 1.8
        depth_coeff = 2.6
        dropout_rate = 0.5
        super(EfficientNetB6, self).__init__(
            width_coefficient=width_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            include_top=include_top,
            pooling=pooling,
            classes=classes,
            **kwargs)

class EfficientNetB7(EfficientNet):
    def __init__(self, 
                 include_top=True,
                 pooling: str = None,
                 classes: int = 1000,
                 **kwargs):
        width_coeff = 2.0
        depth_coeff = 3.1
        dropout_rate = 0.5
        super(EfficientNetB7, self).__init__(
            width_coefficient=width_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            include_top=include_top,
            pooling=pooling,
            classes=classes,
            **kwargs)