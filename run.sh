#!/bin/bash

data_dir=/home/kmyoon/data/LibriSpeech
work_dir=/home/kmyoon/exp/tf-asr

#data_dir=/home/feesh/corpus/LibriSpeech
#work_dir=/home/feesh/projects/tf-asr

out_dir=${work_dir}/outputs/librispeech
mkdir -p ${out_dir}

config=${work_dir}/configs/librispeech_conformer_v1.yml

train_tfrecord=${out_dir}/train.tfrecord
valid_tfrecord=${out_dir}/valid.tfrecord

# Data preparation
if [ ! -f ${train_tfrecord} ] || [ ! -f ${valid_tfrecord} ]; then
    python3 prepare_librispeech.py \
        --data_dir ${data_dir} \
        --out_dir ${out_dir} \
        --config ${config} \
        || exit 1
fi

# Train
python3 temp.py \
    --train_tfrecord ${valid_tfrecord} \
    --valid_tfrecord ${valid_tfrecord} \
    --config ${config} \
    --log_dir ${out_dir}/logs \
    --chkpt_dir ${out_dir}/chkpts \
    || exit 1