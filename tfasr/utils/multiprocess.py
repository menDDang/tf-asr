from __future__ import absolute_import
from ast import arg
from concurrent.futures import process
from multiprocessing.dummy import Process

from typing import Callable, List
import abc
import argparse
import multiprocessing
import os
import signal
import types

from rich import traceback

from tfasr.utils import logging

traceback.install()
#LOG = logging.get_detail_logger(__name__, multi=True)
LOG = logging.DetailLogger(__name__, multi=True)


class ForceRaiseError(Exception):
    """Custom error."""

class ProcessBase(multiprocessing.Process):
    def __init__(self, 
                 index: int,
                 num_total_processes: int,
                 queue_ref: multiprocessing.JoinableQueue,
                 counter_ref: multiprocessing.Value,
                 **kwargs):
        super(ProcessBase, self).__init__(**kwargs)
        self.process_idx = index
        self.num_total_processes = num_total_processes
        self.queue = queue_ref
        self.counter = counter_ref
        
    @abc.abstractmethod
    def run(self):
        raise NotImplementedError()
        
class MultiProcessManager(object):
    """ Multi processing controller.
    
    Instantiate multiple classes that inherits ProcessBase.
    
    """
    def __init__(self, 
                 process_class: ProcessBase,
                 num_processes: int,
                 **kwargs) -> None:
        """ Init the manager and check validity.
        
        Raises:
            TypeError: Raised when process is
                not instance of ProcessBase.
            AttributeError: Raised when ClassWithRun does
                not have .run() method.
            TypeError: Raised when arguments is
                not instance of argparse.Namespace.
            TypeError: Raised when num_processes is
                not instance of int.
            NotImplementedError: Raised when process's method
                is not implemented.
        """
        
        if not issubclass(process_class, ProcessBase):
            LOG.error(
                "{} has wrong type, must be instance of ProcessBase."
                .format(process_class))
            raise TypeError
        if not isinstance(num_processes, int):
            LOG.error("{} has wrong type, must be instance of int.".format(
                num_processes))
            raise TypeError
        
        self.process_class = process_class
        self.num_processes = num_processes
        
        
        self.queue = multiprocessing.JoinableQueue()
        self.counter = multiprocessing.Value("i", lock=True)
        self._processes = []
        for process_idx in range(self.num_processes):
            self._processes.append(
                self.process_class(
                    index=process_idx, 
                    num_total_processes=self.num_processes,
                    queue_ref=self.queue,
                    counter_ref=self.counter,
                    **kwargs))
        signal.signal(signal.SIGINT, MultiProcessManager._signal_handler)
        signal.signal(signal.SIGTERM, MultiProcessManager._signal_handler)
        
    @property
    def processes(self):
        """ Processes getter, indirect. """
        return self._get_processes()
        
    def join(self) -> None:
        """ Join the running processes for graceful shutdown. """
        for p in self._processes:
            p.join()
        LOG.info("All processes were joined.")
        
    def start(self) -> None:
        """ Spawn processes for future scheduling. """
        for p in self._processes:
            p.start()
        LOG.info("All processes were spawned.")
        
    def _set_process(self, processes: List[type]) -> None:
        """ Processes setter, direct. """
        self._processes = processes
        
    def _get_processes(self):
        """ Processes getter, direct"""
        return self._processes
        
    @staticmethod
    def _signal_handler(signum: signal.Signals,
                        frame: types.FrameType) -> None:
        """Raise ForceRaiseError on each and every main thread.

        In main process, joining should be called upon ForceRaiseError, and
        in other processes, cleaning up sould kick in upon ForceRaiseError.

        Args:
            signum: signal.Signals, interruptive signal id.
            frame: types.FrameType, stack frame at the moment.

        Returns:

        Raises:
            ForceRaiseError: Raised always.
        """
        LOG.info("Caught signal {} on PID {}, joining processes...".format(
            signum, os.getpid()))
        raise ForceRaiseError
        
        
def launch_multi_process(process_class: ProcessBase,
                         num_processes: int,
                         **kwargs) -> None:
    
    """Spawn and control multi processor(s).

    Using, if any,
    .start() & .run() for spawning,
    .processes for referencing,
    .join() for coordinating.

    Args:
        process: utils.multiprocess.ProcessBase,
            class that has .run() to be parallelized.
        arguments: argparse.Namespace, arguments to ClassWithRun.
        num_processes: int, scalar[], how many processes to spawn.

    Returns:

    Raises:
        cprocess.ForceRaiseError: Raised when interuptive signal inbound
            is caught by cprocess._signal_handler.
    """
    manager = MultiProcessManager(process_class=process_class,
                                  num_processes=num_processes,
                                  **kwargs)
    try:
        manager.start()
        for p in manager.processes:
            p.join()
    except ForceRaiseError:
        manager.join()