from __future__ import absolute_import

import abc
import unicodedata
from typing import Dict, List

import tensorflow as tf

class TextTokenizerBase(metaclass=abc.ABCMeta):
    """ Base class for text tokenization """

    def __init__(self, eos: int = 0, sos: int = 1):
        self.eos = eos
        self.sos = sos
        self.max_length = None

    @property
    def shape(self):
        return [self.max_length]
    
    def update_length(self, new_length: int):
        if self.max_length is None:
            self.max_length = new_length
        else:
            self.max_length = max(self.max_length, new_length)

    def reset_length(self):
        self.max_length = None

    def prepand_sos(self, tokens):
        return tf.concat([[self.sos], tokens], axis=0)

    def append_eos(self, tokens):
        return tf.concat([tokens, [self.eos]], axis=0)

    @abc.abstractclassmethod
    def tokenize(self, text: str):
        """
        Convert text to sequence of indices
        """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def detokenize(self, indices):
        """
        Convert sequnce of indices to text
        """
        raise NotImplementedError()
