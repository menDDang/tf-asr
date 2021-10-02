import argparse
import os
import datetime

import tensorflow as tf

import utils
from datasets.librispeech_dataset import LibrispeechDataset
from models.seq2seq import Seq2Seq
from models.seq2seq_lstm import Seq2SeqLstm
from models.seq2seq_tds_conv import Seq2SeqTdsConv
from models.seq2seq_tcr import Seq2SeqTcr


UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 1
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 2
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 3
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 0  # unused, by default


def buildModel(hp, vocab_size, max_output_length):
    sos = 2
    eos = 3
    if hp["model_type"] == "seq2seq_lstm":
        hp["lstm_net"] = {
            "encoder_conv_filters" : 64,
            "encoder_conv_kernel_size" : 8,
            "encoder_lstm_num_layers" : 8,
            "encoder_lstm_units" : 512,
            "encoder_dropout_prob" : 0.0,
            "decoder_lstm_units" : 1024,
            "decoder_dropout_prob" : 0.0
        }
        model = Seq2SeqLstm(hp, vocab_size, max_output_length=max_output_length, sos=sos, eos=eos)
    elif hp["model_type"] == "seq2seq_tds11":
        hp["tds_conv_net"] = {
            "kernel_size" : 21,
            "num_tds_blocks_1" : 2,
            "num_channels_1" : 10,
            "num_tds_blocks_2" : 3,
            "num_channels_2" : 14,
            "num_tds_blocks_3" : 6,
            "num_channels_3" : 16,
            "encoder_dense_units" : 1024
        }
        model = Seq2SeqTdsConv(hp, vocab_size, max_output_length=max_output_length, sos=sos, eos=eos)
    elif hp["model_type"] == "seq2seq_tcr8":
        hp["tcr_net"] = {
            "kernel_size" : 9,
            "num_tcr_blocks" : 8,
            "num_channels" : 64,
        }
        model = Seq2SeqTcr(hp, vocab_size, max_output_length=max_output_length, sos=sos, eos=eos)
    else:
        raise ValueError("Invalid type of model")
    return model

def buildOptimizer(hp):
    #lr = tf.optimizers.schedules.PolynomialDecay(
    #    hp["init_learning_rate"],
    #    hp["learning_rate_decay_steps"],
    #    hp["end_learning_rate"])
    lr = hp["init_learning_rate"]

    if hp["optimizer"] == "sgd":
        optimizer = tf.optimizers.SGD(lr)
        return optimizer
    elif hp["optimizer"] == "adam":
        optimizer = tf.optimizers.Adam(lr)
        return optimizer
    elif hp["optimizer"] == "adadelta":
        optimizer = tf.optimizers.Adadelta(lr, rho=0.95, epsilon=1e-10)
        return optimizer
    else:
        raise ValueError("Invalid type of optimizer")

@tf.function
def compute_ctc_loss(y_pred, y_true, y_pred_length, y_true_length, blank=0, eos=2):
    ''' 
    y_pred: tensor of shape [batch_size, frames, num_labels].
    y_true: tensor of shape [batch_size, max_label_seq_length] or SparseTensor
    y_pred_length: tensor of shape [batch_size]
    y_true_length: tensor of shape [batch_size]
    '''
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=y_true_length,
        logit_length=y_pred_length,
        blank_index=blank,
        logits_time_major=False
    )
    loss = tf.reduce_mean(loss)
    return loss

@tf.function
def compute_cer(y_pred, y_true, y_pred_length):
    y_pred = tf.transpose(y_pred, [1, 0, 2])
    y_pred, _ = tf.nn.ctc_greedy_decoder(
        y_pred, y_pred_length, merge_repeated=True
    )
    y_pred = tf.cast(y_pred[0], tf.int32)
    cer = tf.reduce_mean(tf.edit_distance(
        hypothesis=y_pred,
        truth=tf.sparse.from_dense(y_true),
        normalize=True
    ))
    return cer

@tf.function
def train_step(x, x_length, y_true, y_true_length, model, optimizer, clip_norm=15.0, apply_soft_window=False):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True, y_true=y_true, apply_soft_window=apply_soft_window)
        loss = compute_ctc_loss(y_pred, y_true, y_true_length, y_true_length)
        
    vars = model.trainable_variables
    grads = tape.gradient(loss, vars)
    if clip_norm > 0:
        clipped_grads = []
        for grad in grads:
            clipped_grads.append(tf.clip_by_norm(grad, clip_norm))
        optimizer.apply_gradients(zip(clipped_grads, vars))
    else:
        optimizer.apply_gradients(zip(grads, vars))
    cer = compute_cer(y_pred, y_true, y_true_length)

    return loss, cer

@tf.function
def eval_step(x, x_length, y_true, y_true_length, model):
    y_pred = model(x, training=False)
        
    loss = compute_ctc_loss(y_pred, y_true, y_true_length, y_true_length)
    cer = compute_cer(y_pred, y_true, y_true_length)
    return loss, cer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--chkpt_dir", type=str, default="./chkpt")
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--valid_dataset_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parseArguments()

    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    log_dir = os.path.join(args.log_dir, current_time)
    chkpt_dir = os.path.join(args.chkpt_dir, current_time)

    config = utils_load_yaml(args.config)

    libri_train = LibriSpeechDataset(hp)
    libri_train.load(args.train_dataset_path)
    train_dataset = libri_train.dataset

    libri_valid = LibriSpeechDataset(hp)
    libri_valid.load(args.valid_dataset_path)
    valid_dataset = libri_valid.dataset
    
    if max_mel_length > 0 and max_token_length > 0:
        train_dataset = train_dataset.filter(lambda x, x_length, y_true, y_length : (
            tf.shape(x)[0] < max_mel_length 
            and tf.shape(y_true)[0] < max_token_length
        ))
        valid_dataset = valid_dataset.filter(lambda x, x_length, y_true, y_length : (
            tf.shape(x)[0] < max_mel_length and tf.shape(y_true)[0] < max_token_length
        ))

    train_dataset = train_dataset.padded_batch(
        batch_size=hp["batch_size"], 
        padded_shapes=([None, num_mels], [], [None], [])
    )
    valid_dataset = valid_dataset.padded_batch(
        batch_size=hp["batch_size"], 
        padded_shapes=([None, num_mels], [], [None], [])
    )
    train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    # Build model
    model = buildModel(hp, libri_train.vocab_size, max_output_length=max_token_length)
    if model == None:
        raise ValueError("model is None")

    # Build optimizer
    optimizer = buildOptimizer(hp)

    # Train
    for iter in range(1, hp["num_iter"] + 1):
        train_loss_object = tf.keras.metrics.Mean()
        train_cer_object = tf.keras.metrics.Mean()
        if iter < hp["num_soft_window_pretraining_epochs"]: 
            apply_soft_window = True
        else:
            apply_soft_window = False
        for step, (x, x_length, y_true, y_true_length) in enumerate(train_dataset):
            y_pred = model(x, training=True, y_true=y_true, apply_soft_window=apply_soft_window)
            train_loss, train_cer = train_step(
                x=x, 
                x_length=x_length, 
                y_true=y_true, 
                y_true_length=y_true_length, 
                model=model, 
                optimizer=optimizer, 
                apply_soft_window=apply_soft_window, 
                clip_norm=hp["grad_clip_norm"])
            train_loss_object.update_state(train_loss)
            train_cer_object.update_state(train_cer)
            print("Step : %d, Train Loss : %f, Train CER : %f" % (step, float(train_loss), float(train_cer)))
            #if step == 100:
            #    break
        train_loss = train_loss_object.result()
        train_cer = train_cer_object.result()
        
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar("Train Loss", train_loss, step=iter)
            tf.summary.scalar("Train CER", train_cer, step=iter)
        
        valid_loss_object = tf.keras.metrics.Mean()
        valid_cer_object = tf.keras.metrics.Mean()
        for step, (x, x_length, y_true, y_true_length) in enumerate(valid_dataset):
            valid_loss, valid_cer = eval_step(x, x_length, y_true, y_true_length, model)
            valid_loss_object.update_state(valid_loss)
            valid_cer_object.update_state(valid_cer)
            #if step == 5:
            #    break
        valid_loss = valid_loss_object.result()
        valic_cer = valid_cer_object.result()

        print("Iter : %d, Train Loss : %f, Train CER : %f, Valid Loss : %f, Valid CER : %f" \
            % (iter, float(train_loss), float(train_cer), float(valid_loss), float(valid_cer)))

        #continue
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar("Valid Loss", valid_loss, step=iter)
            tf.summary.scalar("Valid CER", valid_cer, step=iter)

        # Save checkpoint
        chkpt_file_path = os.path.join(chkpt_dir, "chkpt_step:{}_loss:{:.4f}".format(iter, float(train_loss)))
        model.save_weights(chkpt_file_path)        
        