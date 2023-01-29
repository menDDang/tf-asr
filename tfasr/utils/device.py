from __future__ import absolute_import

from typing import Dict, List

import tensorflow as tf


def set_devices(devices: List[int], cpu: bool = False):
    if cpu:
        cpus = tf.config.list_physical_devices("CPU")
        tf.config.set_visible_devices(cpus, "CPU")
        tf.config.set_visible_devices([], "GPU")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            visible_gpus = [gpus[i] for i in devices]
            tf.config.set_visible_devices(visible_gpus, "GPU")