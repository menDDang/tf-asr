from __future__ import absolute_import

import abc
import os
from typing import List, Tuple

from rich import traceback
import numpy as np
import tensorflow as tf

from tfasr import utils

traceback.install()
LOG = utils.logging.DetailLogger(__name__, multi=True)


BYTES_PER_TFRECORDS = int(1.5e+8)  # How much records to stack to a .tfrecords.


class DataProcessor(utils.multiprocess.ProcessBase):

    def __init__(self,
                 total_entry: List[str],
                 outdir: str,
                 **kwargs) -> None:
        super(DataProcessor, self).__init__(**kwargs)
        self.total_entry = total_entry
        self.outdir = outdir


    @abc.abstractmethod
    def preprocess(self, one_line_in_entry) -> Tuple[tf.Tensor]:
        raise NotImplementedError("DataProcessor::preprocess() is not implemented")


    @abc.abstractmethod
    def flush_to_pipe(self, pipe, file, batch):
        raise NotImplementedError("DataProcessor::flosh_to_pipe() is not implemented")


    def run(self) -> None:
        """ Parallel execution body. 
        
        When spawned by multiprocess.MultiProcessManager, this method will
        be parallelized on respective process' thread.
        
        Args:

        Returns:

        Raises:
            multiprocess.ForceRaiseError: Raised when interruptive signal inbound
                is caught by multiprocess._signal_handler.
            AssertionError: Raised when there exists no lock or
                there exists trial to release other process' lock.
        """
        
        os.makedirs(self.outdir, exist_ok=True)
        
        try:
            LOG.info(f"Data process {self.process_idx} started.")
            if self._check_previous_jobs():
                return
            entry = self._distribute_entry()
            if entry is None:
                LOG.info("There is no jobs to do.")
                return
            LOG.info(f"Num entry: {len(entry)}")

            # Prepare initial job
            self.queue.put(1, True)
            total_cap = len(entry)
            with self.counter.get_lock():
                self.counter.value += 1
                out_file_idx = self.counter.value
            file = os.path.join(self.outdir, "{}.tfrecords".format(str(out_file_idx).zfill(10)))
            pipe = tf.io.TFRecordWriter(file)
            progress_idx = 0
            LOG.progress(current_iter=progress_idx,
                         total_iter=total_cap,
                         prefix="Process {} progress:".format(self.process_idx))
                 
                 # Start        
            while entry:
                try:
                    batch = self.preprocess(entry.pop())
                    (pipe, file) = self.flush_to_pipe(pipe, file, batch)
                    progress_idx += 1
                    LOG.progress(current_iter=progress_idx,
                                total_iter=total_cap,
                                prefix="Process {} progress:".format(self.process_idx))
                #except IndexError:
                #    progress_idx += 1
                #    continue
                #except TypeError:
                #    progress_idx += 1
                #    continue
                #except ValueError:
                #    progress_idx += 1
                #    continue
                except AssertionError:
                    progress_idx += 1
                    continue
            
            # Must close, since last .tfrecords' size
            # could be below BYTES_PER_TFRECORDS.
            pipe.close()
            
            # Finish.
            self.queue.get(True)
            self.queue.task_done()
            LOG.info("Data process {} waiting other(s).".format(
                self.process_idx))
            self.queue.join()
            LOG.info("Data process {} finished.".format(
                self.process_idx))
                
        # Terminate.
        except utils.multiprocess.ForceRaiseError:
            LOG.info("Data process {} cleaning up.".format(
                self.process_idx))
            self.queue.cancel_join_thread()
            try:
                self.counter_0.get_lock().release()
            except AssertionError:
                pass
            LOG.info("Data process {} terminated.".format(
                self.process_idx))
            
    def _check_previous_jobs(self):
        if os.path.isfile(
            os.path.join(self.outdir, 
                         "{}.tfrecords".format(str(self.num_total_processes).zfill(10)))):
            LOG.info(f"Data process {self.process_idx} found previously processed data, escaping...")
            return True
        else:
            return False
    
    def _distribute_entry(self):
        len_entry = len(self.total_entry)
        plus_minus = 1
        allocated_idx = (np.arange(0, len_entry) + plus_minus) * (
            (np.arange(0, len_entry) // 1 % self.num_total_processes) == self.process_idx)
        allocated_idx = allocated_idx[allocated_idx.nonzero()] - plus_minus
        if allocated_idx.size == 0:
            LOG.info(
                "Data process {} has not been allocated any data for the processing, escaping..."
                .format(self.process_idx))
            return
        entry = [self.total_entry[idx] for idx in allocated_idx]
        return entry