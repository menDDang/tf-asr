from __future__ import absolute_import

import abc
from typing import Any, Dict

DEFAULT_BUFFER_SIZE = 100

class DatasetBase(metaclass=abc.ABCMeta):
    """ Base class of dataset """

    def __init__(self, 
                 num_cpus: int = 4,
                 use_cache: bool = False,
                 shuffle: bool = False,
                 buffer_size: int = DEFAULT_BUFFER_SIZE,
                 drop_remainder: bool = True) -> None:
        self.num_cpus = num_cpus
        self.use_cache = use_cache
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        if self.buffer_size <= 0 and self.shuffle:
            raise ValueError("buffer_size must be positive when shuffle is on")
        self.drop_remainder = drop_remainder
        
    @abc.abstractmethod
    def create(self, *args, **kwargs) -> None:
        """ Run all data processings. """
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, file_path) -> None:
        """ Load from disk. """
        raise NotImplementedError()
