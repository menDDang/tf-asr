from __future__ import absolute_import

from typing import Dict
import tensorflow as tf

from .tokenizer_base import TextTokenizerBase

ENGLISH_CHARACTER_LIST = [
    " ", "a", "b",
    "c", "d", "e", 
    "f", "g", "h", 
    "i", "j", "k", 
    "l", "m", "n", 
    "o", "p", "q", 
    "r", "s", "t", 
    "u", "v", "w", 
    "x", "y", "z", 
    "'"
]

class CharTokenizer(TextTokenizerBase):
    def __init__(self, config: Dict, characters=ENGLISH_CHARACTER_LIST, **kwargs):
        super(CharTokenizer, self).__init__(**kwargs)
        self.config = config
        self.num_classes = len(characters) + 2  # ENGLISH_CHARACTER_LIST + <SOS> + <EOS>
        
        keys = tf.constant(characters)
        values = tf.constant(range(2, self.num_classes))       # 0 : <EOS>, 1 : <SOS>, ...
        self.char_to_index_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1
        )
        
    def tokenize(self, text):
        text = tf.strings.strip(text)           # erase '\n'
        text = tf.strings.lower(text)           # normalize to lower charcters
        text = tf.strings.bytes_split(text)
        return self.char_to_index_table.lookup(text)

    def detokenize(self, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            indices = self.normalize_indices(indices)
            tokens = tf.gather_nd(
                params=self.tokens,
                indices=tf.expand_dims(indices, axis=-1))
            tokens = tf.strings.reduce_join(tokens, axis=-1)
        return tokens
