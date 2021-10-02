import os
import math
import argparse

import tensorflow as tf
from tensorflow.python.keras.backend_config import epsilon
from datasets.librispeech_dataset import LibrispeechDataset

import utils


class ConformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, max_lr=None):
        super(ConformerSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # lr = (d_model^-0.5) * min(step^-0.5, step*(warm_up^-1.5))
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        if self.max_lr is not None:
            return tf.math.minimum(self.max_lr, lr)
        return lr

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
        }

def build_optimizer(config : dict, conformer=None):

    if config["learning_rate_type"] == "fixed":
        learning_rate = config["init_lr"]
    elif config["learning_rate_type"] == "conformer":
        learning_rate = ConformerSchedule(
            d_model=conformer.dmodel,
            warmup_steps=config.get("warmup_steps", 10000),
            max_lr=(0.05 / math.sqrt(conformer.dmodel))
        )
    else:
        raise ValueError("Invalid learning rate type")

    if config["optimizer_type"] == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate,
            beta_1=config.get("beta_1",0.9),
            beta_2=config.get("beta_2", 0.999),
            epsilon=config.get("epsilon", 1e-7),
            amsgrad=config.get("amsgrad", False)
        )
    else:
        raise ValueError("Invalid optimizer type")

    return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="The file path of model configuration file")
    parser.add_argument("--devices", type=int, nargs="*", default=[0], help="Devices' ids to apply distributed training")

    args = parser.parse_args

    config = utils.load_yaml(args.config) 
    model_train_hp = config["model_train_config"]

    strategy = env_util.setup_strategy(args.devices)

    libri_train = LibrispeechDataset("", config)
    libri_valid = LibrispeechDataset("", config)

    global_batch_size = args.bs or config.learning_config.running_config.batch_size
    global_batch_size *= strategy.num_replicas_in_sync

    with strategy.scope():
        # build model
        #conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
        #conformer.make(
        #    speech_featurizer.shape,
        #    prediction_shape=text_featurizer.prepand_shape,
        #    batch_size=global_batch_size
        #)
        #if args.pretrained:
        #    conformer.load_weights(args.pretrained, by_name=True, skip_mismatch=True)
        #conformer.summary(line_length=100)
        optimizer = build_optimizer(config)
        conformer.compile(
            optimizer=optimizer,
            experimental_steps_per_execution=args.spx,
            global_batch_size=global_batch_size,
            blank=text_featurizer.blank
        )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config.running_config.checkpoint),
        tf.keras.callbacks.experimental.BackupAndRestore(config.learning_config.running_config.states_dir),
        tf.keras.callbacks.TensorBoard(**config.learning_config.running_config.tensorboard)
    ]

    conformer.fit(
        train_data_loader,
        epochs=config.learning_config.running_config.num_epochs,
        validation_data=eval_data_loader,
        callbacks=callbacks,
        steps_per_epoch=train_dataset.total_steps,
        validation_steps=eval_dataset.total_steps if eval_data_loader else None
    )