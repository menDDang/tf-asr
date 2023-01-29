# The MIT License (MIT)
# Copyright (c) 2020 Yoon, Hyebin & Kim, Wiback

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.
"""This module includes custom logging utilities. """

from __future__ import absolute_import
from __future__ import print_function

import logging
import multiprocessing
from rich import traceback

traceback.install()

def get_detail_logger(name: str, multi: bool = False) -> logging.Logger:
    if multi:
        logger = multiprocessing.get_logger()
        logger.setLevel(logging.INFO)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
    logging_format = logging.Formatter(
        "%(asctime)-25s"
        "%(levelname)-10s"
        "%(filename)s:%(lineno)-20d"
        "%(process)d:%(thread)-20d"
        "%(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging_format)
    logger.addHandler(stream_handler)
    return logger
    
class DetailLogger(object):
    """Preset logging module's logger.
    
    Attributes:
        _multi: bool, if true, multi processing compatible logger kicks in.
        _nib: logging.Logger, the single or multi processing logger.
    """
    
    def __init__(self,
                 name: str, 
                 multi: bool = False) -> None:
        self._multi = multi
        
        if self._multi:
            self._nib = multiprocessing.get_logger()
            self._nib.setLevel(logging.INFO)
        else:
            self._nib = logging.getLogger(name)
            self._nib.setLevel(logging.INFO)
            
        if not self._nib.handlers:
            logging_format = logging.Formatter(
                "%(asctime)-25s"
                "%(levelname)-10s"
                "%(filename)s:%(lineno)-20d"
                "%(process)d:%(thread)-20d"
                "%(message)s"
            )
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging_format)
            self._nib.addHandler(stream_handler)
            
    def debug(self, message: str) -> None:
        self._nib.debug(message)
        
    def info(self, message: str) -> None:
        self._nib.info(message)
    
    def warning(self, message: str) -> None:
        self._nib.warning(message)
    
    def error(self, message: str) -> None:
        self._nib.error(message)
    
    def critical(self, message: str) -> None:
        self._nib.critical(message)
    
    def progress(self,
                current_iter,
                total_iter,
                prefix="Progress",
                suffix="Complete",
                decimals=2,
                total_length=50):
        """ Terminal progress bar to stdout. 
        
        Args:
            current_iter: int, current iteration count.
            total_iter: int, total iteration count.
            prefix: str, prefix string.
            suffix: str, suffix string.
            decimals: int, positive number of decimals in percent.
            total_length: int, leng of bar.
        """
        
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (current_iter / float(total_iter)))
        filled_length = int(total_length * current_iter // total_iter)
        bars = ("█" * filled_length) + ("-" * (total_length - filled_length))
        self._nib.info(f"\r{prefix} |{bars}| {percent}% {suffix}")
        if current_iter == total_iter:
            self._nib.info("\n")