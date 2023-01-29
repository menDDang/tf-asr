from __future__ import absolute_import

import os
import collections
import unicodedata
from typing import List

import tensorflow as tf
import sentencepiece as sp

from tfasr.data.text.tokenizer_base import TextTokenizerBase




class SentencePieceTokenizer(TextTokenizerBase):
    """
    Extract text feature based on sentence piece package.
    """

    TextToken = collections.namedtuple('TextToken', ['name', 'id'])
    PAD_TOKEN = TextToken(name="<pad>",  id=0)  # unused, by default
    UNK_TOKEN = TextToken(name="<unk>",  id=1)  # Un-known token
    BOS_TOKEN = TextToken(name="<s>",    id=2)  # Begin of Speech token
    EOS_TOKEN = TextToken(name="</s>",   id=3)  # End of Speech token
    
    def __init__(self, 
                 model: sp.SentencePieceProcessor = None,
                 **kwargs):
        super(SentencePieceTokenizer, self).__init__(
            sos=self.BOS_TOKEN.id, 
            eos=self.EOS_TOKEN.id, 
            **kwargs)
        self.model = model
        self.blank = self.PAD_TOKEN.id  # treats blank as 0 (pad)
        self.num_classes = self.model.get_piece_size()
        self._init_vocabulary()

    def _init_vocabulary(self):
        self.tokens = []
        for idx in range(1, self.num_classes):
            self.tokens.append(self.model.decode_ids([idx]))
        self.non_blank_tokens = self.tokens.copy()
        self.tokens.insert(0, "")
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8")
        self.upoints = self.upoints.to_tensor()  # [num_classes, max_subword_length]

    @classmethod
    def build_from_corpus(cls,
                          output_path_prefix: str,
                          entry: List,
                          vocab_size: int = 10000,
                          num_threads: int = 4):
        def _corpus_iterator():
            _entry = entry[1:] # erase first line of entry
            for line in _entry:
                _, transcript = line.split("\t")
                transcript = unicodedata.normalize("NFC", transcript.lower())
                transcript = transcript.strip("\n")
                yield transcript
                
        # Train SentencePiece Model

        sp.SentencePieceTrainer.Train(
            sentence_iterator=_corpus_iterator(),
            model_prefix=output_path_prefix,
            vocab_size=vocab_size,
            num_threads=num_threads,
            unk_id=cls.UNK_TOKEN.id,
            bos_id=cls.BOS_TOKEN.id,
            eos_id=cls.EOS_TOKEN.id,
            pad_id=cls.PAD_TOKEN.id,
            unk_surface="__UNKNOWN__")
            
        processor = sp.SentencePieceProcessor()
        processor.Load(output_path_prefix + ".model")
        vocab = {
            i : processor.IdToPiece(i) for i in range(processor.GetPieceSize())
        }
        vocab = {
            i: s for i, s in vocab.items() if s not in { 
                cls.UNK_TOKEN.name, 
                cls.BOS_TOKEN.name, 
                cls.EOS_TOKEN.name, 
                cls.PAD_TOKEN.name
            }
        }
        with open(output_path_prefix + ".txt", "w") as f:
            for _, s in sorted(vocab.items(), key=lambda x: x[0]):
                f.write(f"{s} 1\n")

        return cls(processor)

    @classmethod
    def load_from_file(cls, model_file_path: str):
        processor = sp.SentencePieceProcessor()
        processor.load(model_file_path)
        return cls(model=processor)

    def tokenize(self,
                 text: str) -> tf.Tensor:
        """ Convert string to a list of integers
        # encode: text => id
        sp.encode_as_pieces('This is a test') --> ['▁This', '▁is', '▁a', '▁t', 'est']
        sp.encode_as_ids('This is a test') --> [209, 31, 9, 375, 586]
        Args:
            text: string (sequence of characters)
        Returns:
            sequence of ints in tf.Tensor
        """
        #text = self.normalize_text(text)
        indices = self.model.encode_as_ids(text)
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def detokenize(self,
                   indices: tf.Tensor) -> tf.Tensor:
        """ Convert list of indices to string
        # decode: id => text
        sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']) --> This is a test
        sp.decode_ids([209, 31, 9, 375, 586]) --> This is a test
        Args:
            indices: tf.Tensor with dim [B, None]
        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        indices = self.normalize_indices(indices)
        with tf.device("/CPU:0"):  # string data is not supported on GPU

            def decode(x):
                if x[0] == self.blank:
                    x = x[1:]
                return self.model.decode_ids(x.tolist())

            text = tf.map_fn(
                lambda x: tf.numpy_function(decode, inp=[x], Tout=tf.string),
                indices,
                fn_output_signature=tf.TensorSpec([], dtype=tf.string),
            )
        return text

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def indices2upoints(
        self,
        indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]
        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        with tf.name_scope("indices2upoints"):
            indices = self.normalize_indices(indices)
            upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
            return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))