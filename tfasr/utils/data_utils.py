from __future__ import absolute_import
import tensorflow as tf

def create_inputs(
    inputs: tf.Tensor,
    input_lengths: tf.Tensor,
    predictions=None,
    prediction_lengths=None,
) -> dict:
    data = {
        "inputs": inputs,
        "input_lengths": input_lengths,
    }
    if predictions is not None:
        data["predictions"] = predictions
        data["prediction_lengths"] = prediction_lengths
    return data


def create_logits(
    logits: tf.Tensor,
    logit_lengths: tf.Tensor,
) -> dict:
    return {"logits": logits, "logit_lengths": logit_lengths}


def create_labels(
    labels: tf.Tensor,
    label_lengths: tf.Tensor,
) -> dict:
    return {
        "labels": labels,
        "label_lengths": label_lengths,
    }