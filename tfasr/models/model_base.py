from __future__ import absolute_import

import abc
from typing import List
import tensorflow as tf


class ModelBase(tf.keras.Model):

    def __init__(self, **kwargs):
        super(ModelBase, self).__init__(**kwargs)
        self._metric_dict = dict()
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
    

    def add_metric(self, metric: tf.keras.metrics.Metric):
        self._metric_dict[metric.name] = metric


    @property
    def metrics(self) -> List:
        return [m for m in self._metric_dict.values() ]
        

    def compile(self, loss, optimizer, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, run_eagerly=False, **kwargs)


    def train_step(self, data):
        """
        Args:
            data : A `list` containing inputs and y_true, where
                inputs : Dict containing a batch of training data.
                y_true : Dict containing a batch of labels.
                
        Returns:
            A `dict` containing values that will be passed to 
            tf.keras.callbacks.CallbackList.on_train_batch_end. 
            Typically, the values of the Model's metrics are returned. 
            Example: {'loss': 0.2, 'accuracy': 0.7}.
        """
        inputs, y_true = data
        
        with tf.GradientTape() as tape:
            #y_pred = self.call(inputs, training=True)
            y_pred = self(inputs, training=True)
            loss = self.loss(y_true, y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_metric.update_state(loss)
        for m in self.metrics:
            m.update_state(y_true, y_pred)
        output = { name: metric.result() for name, metric in self._metric_dict.items() }
        output["loss"] = self.loss_metric.result()
        return output
        

    def test_step(self, data):
        inputs, y_true = data
        #y_pred = self.call(inputs, training=False)
        y_pred = self(inputs, training=False)
        loss = self.loss(y_true, y_pred)

        self.loss_metric.update_state(loss)
        for m in self.metrics:
            m.update_state(y_true, y_pred)
        output = { name: metric.result() for name, metric in self._metric_dict.items() }
        output["loss"] = self.loss_metric.result()
        return output
    
    def infer(self, x, **kwargs):
        raise NotImplementedError()
    