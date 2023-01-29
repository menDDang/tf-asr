import os 
import argparse

import tensorflow as tf

from tfasr.data.datasets.audio_dataset import AudioDataset
from tfasr.data.text.char_tokenizer import CharTokenizer
from tfasr.data.text.sentencepiece_tokenizer import SentencePieceTokenizer
from tfasr.data.audio.feature_extractor import FeatureExtractor

from tfasr.models.tds import TDS
from tfasr.models.efficient_transducer import EfficientTransducer
from tfasr.models.efficient_ctc import EfficientCTC

from tfasr import utils

LOG = utils.get_detail_logger(__name__, multi=False)
NCPUS = 20


class SGDSchedule(tf.keras.callbacks.Callback):
    def __init__(self, 
                 init_lr: float, 
                 decay_rate: float = 0.5, 
                 decay_epoch: int = 40) -> None:            
        super().__init__()
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch

    def on_epoch_begin(self, epoch, logs=None):
        learning_rate = self.init_lr * tf.pow(
            self.decay_rate, tf.cast(epoch // self.decay_epoch, tf.float32) + 1)
        log_msg = "Epoch: {epoch}, Learning Rate: {learning_rate}"
        LOG.info(log_msg.format(epoch=epoch, learning_rate=learning_rate))
        self.model.optimizer.learning_rate = learning_rate


class ADAMSchedule(tf.keras.callbacks.Callback):
    def __init__(self, 
                 init_lr: float, 
                 decay_rate: float = 0.96, 
                 decay_epoch: int = 100) -> None:            
        super().__init__()
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch

    def on_epoch_begin(self, epoch, logs=None):
        # exponential decay
        learning_rate = self.init_lr * tf.pow(
            self.decay_rate, tf.cast(epoch, tf.float32) / self.decay_epoch)
        log_msg = "Epoch: {epoch}, Learning Rate: {learning_rate}"
        LOG.info(log_msg.format(epoch=epoch, learning_rate=learning_rate))
        self.model.optimizer.learning_rate = learning_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
        help="Path of yaml configuration file")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--devices", type=int, nargs="*", default=[0],
        help="GPU devices' ids")
    parser.add_argument("--trained_chkpt_epoch", type=int, default=0)
    args = parser.parse_args()

    # Set devices & strategy
    devices = utils.set_devices(args.devices, cpu=False)
    strategy = tf.distribute.MirroredStrategy(['/gpu:' + str(i) for i in args.devices])
    
    # Configuration
    config = utils.load_yaml(args.config)

    # Get entry files
    train_entry_path = os.path.join(args.outdir, "train_entry.tsv")
    with open(train_entry_path, "r") as f:
        train_entries = f.readlines()
    
    valid_entry_path = os.path.join(args.outdir, "valid_entry.tsv")
    with open(valid_entry_path, "r") as f:
        valid_entries = f.readlines()

    # Load tokenizer
    if config['tokenizer']['type'] == "character":
        LOG.info('Character tokenizer is choosen.')
        tokenizer = CharTokenizer(config=config['tokenizer'])
    elif config['tokenizer']['type'] == "sentencepiece":
        LOG.info('SentencePiece tokenizer is choosen.')
        vocab_size = config["tokenizer"]["vocab_size"]
        tokenizer = SentencePieceTokenizer.load_from_file(
            os.path.join(args.outdir, "sentencepiece", f"sentencepiece-{vocab_size}.model"))

    BATCH_SIZE_PER_DEVICE = config['training']['batch_per_device']
    global_batch_size = BATCH_SIZE_PER_DEVICE * strategy.num_replicas_in_sync
    LOG.info(f"global batch size: {global_batch_size}")
        
    # Load tfrecords
    libri_train = AudioDataset(
        extractor=FeatureExtractor(**config["feature_extractor"], training=True), 
        tokenizer=tokenizer,
        config=config["dataset"])
    libri_train.load(os.path.join(args.outdir, "train"))

    libri_valid = AudioDataset(
        extractor=FeatureExtractor(**config["feature_extractor"], training=False), 
        tokenizer=tokenizer,
        config=config["dataset"])
    libri_valid.load(os.path.join(args.outdir, "valid"))


    libri_train.batch(global_batch_size)
    libri_valid.batch(global_batch_size)


    with strategy.scope():
        # Build model
        model_type = config['model']['model_type']
        assert model_type in ['TDS', 'EFFICIENT_CTC', 'EFFICIENT_TRANSDUCER']

        training_config = config['training']
        if training_config['optimizer'] == 'sgd':
            lr_schedule_callback = SGDSchedule(
                init_lr=training_config["init_lr"],
                decay_rate=training_config["lr_decay_rate"],
                decay_epoch=training_config["lr_decay_epoch"])            
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=training_config["init_lr"],
                clipnorm=training_config["clipnorm"])
        elif training_config['optimizer'] == 'adam':
            lr_schedule_callback = ADAMSchedule(
                init_lr=training_config["init_lr"],
                decay_rate=training_config["lr_decay_rate"],
                decay_epoch=training_config["lr_decay_epoch"])
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=training_config["init_lr"]) 
                #clipnorm=training_config["clipnorm"])

        if model_type == "TDS":
            model = TDS(
                **config["model"]["params"],
                vocab_size=tokenizer.num_classes)
        elif model_type == "EFFICIENT_CTC":
            model = EfficientCTC(
                **config["model"]["params"],
                vocab_size=tokenizer.num_classes)
        elif model_type == "EFFICIENT_TRANSDUCER":
            model = EfficientTransducer(
                **config["model"]["params"],
                vocab_size=tokenizer.num_classes)
        model.compile(optimizer=optimizer, global_batch_size=global_batch_size)
        model.add_metric(utils.CERMetric())
                
        
    # Load checkpoint
    cfg_name = os.path.basename(args.config).split(".")[0]
    checkpoint_path = os.path.join(args.outdir, "chkpts", cfg_name, "epoch-{epoch:08d}")
    if os.path.isfile(checkpoint_path.format(epoch=args.trained_chkpt_epoch) + ".index"):
        print("Load model : ", checkpoint_path.format(epoch=args.trained_chkpt_epoch))
        model.load_weights(checkpoint_path.format(epoch=args.trained_chkpt_epoch))
        initial_epoch = args.trained_chkpt_epoch
    else:
        print("Failed to find model : ", checkpoint_path.format(epoch=args.trained_chkpt_epoch))
        initial_epoch = 0
    
    log_dir = os.path.join(args.outdir, "logs", cfg_name)
    
    callbacks = [
        lr_schedule_callback,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=True,
            mode='auto',
            save_freq='epoch',
            options=None,
            initial_value_threshold=None),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]
    model.fit(
        libri_train.dataset, 
        initial_epoch=initial_epoch,
        epochs=training_config["epochs"],
        validation_data=libri_valid.dataset,
        callbacks=callbacks,
        #steps_per_epoch=1,
        #validation_steps=1
    )

