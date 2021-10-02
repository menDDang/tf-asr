#!/bin/bash

#data_dir=/home/kmyoon/data/LibriSpeech
#work_dir=/home/kmyoon/exp/tf-asr
#out_dir=${work_dir}/outputs/librispeech

data_dir=/home/feesh/corpus/LibriSpeech
work_dir=/home/feesh/projects/tf-asr
out_dir=${work_dir}/outputs/librispeech
mkdir -p ${out_dir}

config=${work_dir}/configs/librispeech_conformer_v1.yml

python3 prepare_librispeech.py \
    --data_dir ${data_dir} \
    --out_dir ${out_dir} \
    --config ${config} \
    || exit 1