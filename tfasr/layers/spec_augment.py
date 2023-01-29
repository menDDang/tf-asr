from __future__ import absolute_import

import tensorflow as tf


@tf.function
def spectrogram_masking(spectrogram, dim=1, masks_number=2, mask_max_size=5):
    """Spectrogram masking on frequency or time dimension.
    Args:
        spectrogram: Input spectrum [batch, time, frequency]
        dim: dimension on which masking will be applied: 1 - time; 2 - frequency
        masks_number: number of masks
        mask_max_size: mask max size
    Returns:
        masked spectrogram
    """

    if dim not in (1, 2):
        raise ValueError('Wrong dim value: %d' % dim)
    input_shape = spectrogram.shape
    time_size, frequency_size = input_shape[1:3]
    dim_size = input_shape[dim]  # size of dimension on which mask is applied
    stripe_shape = [1, time_size, frequency_size]
    for _ in range(masks_number):
        mask_end = tf.random.uniform([], 0, mask_max_size, tf.int32)
        mask_start = tf.random.uniform([], 0, dim_size - mask_end, tf.int32)

        # initialize stripes with stripe_shape
        stripe_ones_left = list(stripe_shape)
        stripe_zeros_center = list(stripe_shape)
        stripe_ones_right = list(stripe_shape)

        # update stripes dim
        stripe_ones_left[dim] = dim_size - mask_start - mask_end
        stripe_zeros_center[dim] = mask_end
        stripe_ones_right[dim] = mask_start

        # generate mask
        mask = tf.concat((
            tf.ones(stripe_ones_left, spectrogram.dtype),
            tf.zeros(stripe_zeros_center, spectrogram.dtype),
            tf.ones(stripe_ones_right, spectrogram.dtype),
        ), dim)
        spectrogram = spectrogram * mask
    return spectrogram


class SpecAugment(tf.keras.layers.Layer):    
    """Spectrogram augmentation.
    
    It is based on paper: SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition https://arxiv.org/pdf/1904.08779.pdf
    """
    def __init__(self, 
                 time_masks_number=2,
                 time_mask_max_size=10,
                 frequency_masks_number=2,
                 frequency_mask_max_size=5,
                 **kwargs):
        super(SpecAugment, self).__init__(**kwargs)

        self.time_mask_max_size = time_mask_max_size
        self.time_masks_number = time_masks_number
        self.frequency_mask_max_size = frequency_mask_max_size
        self.frequency_masks_number = frequency_masks_number

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            # in time dim
            x = spectrogram_masking(
                inputs, 1, self.time_masks_number, self.time_mask_max_size)
            # in frequency dim
            x = spectrogram_masking(
                x, 2, self.frequency_masks_number, self.frequency_mask_max_size)
            return x
        else:
            return tf.identity(inputs)

    def get_config(self):
        config = {
            'frequency_masks_number': self.frequency_masks_number,
            'frequency_mask_max_size': self.frequency_mask_max_size,
            'time_masks_number': self.time_masks_number,
            'time_mask_max_size': self.time_mask_max_size,
        }
        base_config = super(SpecAugment, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

