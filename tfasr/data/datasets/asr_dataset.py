from __future__ import absolute_import

from typing import Dict, List
import os
import unicodedata

from rich import traceback
import tensorflow as tf

from tfasr.data.datasets.data_processor import DataProcessor
from tfasr.data.datasets.dataset_base import DatasetBase
from tfasr.data.text.tokenizer_base import TextTokenizerBase

from tfasr.data.audio import read_audio
from tfasr.data.text import normalize_text
from tfasr import utils

traceback.install()
LOG = utils.logging.DetailLogger(__name__, multi=True)
BYTES_PER_TFRECORDS = int(1.5e+8)  # How much records to stack to a .tfrecords.


class ASRDataProcessor(DataProcessor):
    ''' Automatic Speech Recognition data processor '''

    def __init__(self, total_entry: List[str], outdir: str, **kwargs) -> None:
        super().__init__(total_entry, outdir, **kwargs)

    
    def preprocess(self, one_line_in_entry):
        """ Extract feature from audio && tokenize transcript.
        
        This function is detail implementation of DataProcessor::preprocess()
        ( Inherit from data.DataProcessor)
        """
        (audio_file_path, transcript) = one_line_in_entry
        audio = read_audio(audio_file_path, dtype='float32')
        audio_length = audio.shape[0]
        transcript = normalize_text(transcript)
        transcript_length = len(transcript)
        return (audio, audio_length, transcript, transcript_length)
        
    def flush_to_pipe(self, pipe, file, batch):
        """ Write preprocessed batch into disk
        
        This function is detail implementation of DataProcessor::flush_to_pipe()
        ( Inherit from data.DataProcessor)
        """
        def bytes_features(value):
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value]))
        
        (audio, audio_length, transcript, transcript_length) = batch

        write_feature_map = {
            "audio": bytes_features(
                tf.io.serialize_tensor(audio)),
            "audio_length": bytes_features(
                tf.io.serialize_tensor(audio_length)),
            "transcript": bytes_features(
                tf.io.serialize_tensor(transcript)),
            "transcript_length": bytes_features(
                tf.io.serialize_tensor(transcript_length))
        }
        example = tf.train.Example(features=tf.train.Features(feature=write_feature_map))
        example = example.SerializeToString()
        pipe.write(example)
        if os.stat(file).st_size > BYTES_PER_TFRECORDS:
            pipe.close()
            with self.counter.get_lock():
                self.counter.value += 1
                file_idx = self.counter.value
            file = os.path.join(self.outdir, "{}.tfrecords".format(str(file_idx).zfill(10)))
            pipe = tf.io.TFRecordWriter(file)
        return (pipe, file)

    
class ASRDataSet(DatasetBase):
    def __init__(self, 
                 tokenizer: TextTokenizerBase,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer


    @staticmethod
    def create(total_entry: List,
               out_dir: str,
               num_processes: int = 4) -> None:
        """Create the dataset from entry and save it into disk as TfRecord format.
        
        This function is detail implementation of DataBase::create()
        ( Inherit from data.datasets.DatasetBase )
        """

        utils.multiprocess.launch_multi_process(
            # arguments for MultiProcessManager
            process_class=ASRDataProcessor,
            num_processes=num_processes,
            # auguments for TfRecordGenerator
            total_entry=total_entry,
            outdir=out_dir)
            

    def load(self, datadir: str, max_sample_num: int = None) -> None:
        """ Load pre-created tfrecord format datasets from disk.
        
        This function is detail implementation of DataBase::load()
        ( Inherit from data.datasets.DatasetBase )
        """
        file_pattern = os.path.join(datadir, "*.tfrecords")
        dataset = tf.data.Dataset.list_files(
            file_pattern=file_pattern, 
            shuffle=False)
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x), 
            cycle_length=self.num_cpus)
        
        # dataset : (audio, audio_length, transcript, transcript_length)
        dataset = dataset.map(
            self._parse_example,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        if max_sample_num is not None:
            dataset = dataset.filter(
                lambda audio, audio_length, transcript, transcript_length : audio_length < max_sample_num
            )
        
        # dataset : ({inputs}, {labels})
        dataset = dataset.map(
            self._generate_inputs,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        self.dataset = dataset

        
    def batch(self, batch_size):
        """Batch the datasets and prefetch it. 
        """
        self.dataset = self.dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                utils.create_inputs(
                    inputs=tf.TensorShape([None]),
                    input_lengths=tf.TensorShape([]),
                    predictions=tf.TensorShape([None]),
                    prediction_lengths=tf.TensorShape([])
                ),
                utils.create_labels(
                    labels=tf.TensorShape([None]),
                    label_lengths=tf.TensorShape([])
                )
            ),
            padding_values=(
                utils.create_inputs(
                    #inputs=tf.constant(0, dtype=tf.int16),
                    inputs=tf.constant(0, dtype=tf.float32),
                    input_lengths=0,
                    predictions=tf.constant(0, dtype=tf.int32),
                    prediction_lengths=0
                ),
                utils.create_labels(
                    labels=tf.constant(0, dtype=tf.int32),
                    label_lengths=0
                )
            ),
            drop_remainder=self.drop_remainder
        )

        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)


    def _parse_example(self, serialized_example):
        # map is fixed when written
        read_feature_map = {
            'audio': tf.io.FixedLenFeature([], tf.string),
            'audio_length': tf.io.FixedLenFeature([], tf.string),
            'transcript': tf.io.FixedLenFeature([], tf.string),
            'transcript_length': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(serialized_example, read_feature_map)

        audio = tf.io.parse_tensor(example['audio'], out_type=tf.float32)
        audio_length = tf.io.parse_tensor(example['audio_length'], out_type=tf.int32)
        transcript = tf.io.parse_tensor(example['transcript'], out_type=tf.string)
        transcript_length = tf.io.parse_tensor(example['transcript_length'], out_type=tf.int32)

        return audio, audio_length, transcript, transcript_length


    def _generate_inputs(self, 
                         audio, 
                         audio_length, 
                         transcript, 
                         transcript_length):
        with tf.device("/CPU:0"):
            tokens = tf.py_function(
                lambda x: self.tokenizer.tokenize(x.numpy()),
                inp=[transcript],
                Tout=[tf.int32])[0]
            token_length = tf.gather(tf.shape(tokens), 0)

            inputs = utils.create_inputs(
                inputs=audio,
                input_lengths=audio_length,
                predictions=self.tokenizer.prepand_sos(tokens),
                prediction_lengths=token_length + 1)
            labels = utils.create_labels(
                labels=self.tokenizer.append_eos(tokens),
                label_lengths=token_length + 1
            )
            return inputs, labels