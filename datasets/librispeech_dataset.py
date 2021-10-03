import os
import unicodedata

import tensorflow as tf
import sentencepiece as sp

from .dataset_base import BUFFER_SIZE
from .asr_dataset import AsrDataset



class LibrispeechDataset(AsrDataset):
    def __init__(
        self, 
        sentence_piece_model_path=None,
        config : dict = None
    ):
        super(LibrispeechDataset, self).__init__(config)
    
        if sentence_piece_model_path is not None:
            self.tokenizer = sp.SentencePieceProcessor()
            self.tokenizer.Load(sentence_piece_model_path)

    def tokenize_text(self, text : tf.Tensor):
        text = str(text)
        text = unicodedata.normalize("NFC", text.lower())
        text = text.strip("\n")  # remove trailing newline
        text = text.strip()  # remove trailing space
        indices = self.tokenizer.encode_as_ids(text)
        return tf.convert_to_tensor(indices, dtype=tf.int32)


    def create(self, data_dir, data_types):
        """ Create new dataset from file list """
        # text_list : (file_name)
        txt_list = []
        for data_type in data_types:
            for speaker_id in os.listdir(os.path.join(data_dir, data_type)):
                if speaker_id == '.DS_Store': 
                    continue

                for chapter_id in os.listdir(os.path.join(data_dir, data_type, speaker_id)):
                    if chapter_id == '.DS_Store': 
                        continue

                    file_name = os.path.join(data_dir, data_type, speaker_id, chapter_id, speaker_id + '-' + chapter_id + '.trans.txt')
                    txt_list.append(file_name)

        # file_list : (audio_file_path, transcript)
        file_list = []
        for txt_file_name in txt_list:
            with open(txt_file_name, "r") as f:
                for line in f.readlines():
                    x = line.split('\n')[0].split(' ')
                    uttid = x[0]
                    transcript = ' '.join(x[1:])
                    spkid, chapter_id, _ = uttid.split('-')
                    wav_file_name = os.path.join(data_dir, data_type, spkid, chapter_id, uttid + '.flac')
                    
                    if not os.path.exists(wav_file_name):
                        continue
                    file_list.append([wav_file_name, transcript])
        
        self.dataset = tf.data.Dataset.from_tensor_slices(tf.constant(file_list))
        
        # load audio file & extract features & tokenize transcriptions
        # dataset : (mel, mel_length, label, label_length)
        self.dataset = self.dataset.map(
            lambda line: self.parse(line[0], line[1]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    def parse(self, audio_file_path, transcript):
        
        # load audio files
        audio, sr = tf.py_function(
            lambda path: self.read_audio(path.numpy()),
            inp=[audio_file_path],
            Tout=[tf.float32, tf.int32]
        )
        # extract features
        features = self.extract_feature(audio)

        # tokenize transcript
        tokens = self.tokenize_text(transcript)

        return features, tf.shape(features)[0], tokens, tf.shape(tokens)[0]
        
    def _serializeExample(self, mel, mel_length, tokens, token_length):

        def _bytes_features(value):
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _serialize(_mel, _mel_length, _tokens, _length):
            serialized_mel = tf.io.serialize_tensor(_mel)
            serialized_mel_length = tf.io.serialize_tensor(_mel_length)
            serialized_tokens = tf.io.serialize_tensor(_tokens)
            serialized_token_length = tf.io.serialize_tensor(_length)

            feature = {
                'mel': _bytes_features(serialized_mel),
                'mel_length': _bytes_features(serialized_mel_length),
                'tokens': _bytes_features(serialized_tokens),
                'token_length': _bytes_features(serialized_token_length)
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            example = example.SerializeToString()
            return example

        output = tf.py_function(
            _serialize,
            inp=[mel, mel_length, tokens, token_length],
            Tout=[tf.string])

        return tf.reshape(output, ())


    def _parseExample(self, serialized_example):

        parse_dict = {
            'mel': tf.io.FixedLenFeature([], tf.string),
            'mel_length': tf.io.FixedLenFeature([], tf.string),
            'tokens': tf.io.FixedLenFeature([], tf.string),
            'token_length': tf.io.FixedLenFeature([], tf.string)
        }

        example = tf.io.parse_single_example(serialized_example, parse_dict)

        mel = tf.io.parse_tensor(example['mel'], out_type=tf.float32)
        mel_length = tf.io.parse_tensor(example['mel_length'], out_type=tf.int32)
        tokens = tf.io.parse_tensor(example['tokens'], out_type=tf.int32)
        length = tf.io.parse_tensor(example['token_length'], out_type=tf.int32)
        return (mel, mel_length, tokens, length)


    def save(self, path):
        # Serialize dataset to save as tfrecord format
        self.dataset = self.dataset.map(
            self._serializeExample,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        writer = tf.data.experimental.TFRecordWriter(path, compression_type="ZLIB")
        writer.write(self.dataset)

    def load(self, path):
        self.dataset = tf.data.TFRecordDataset(path, compression_type="ZLIB")
        self.dataset = self.dataset.map(
            self._parseExample,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )