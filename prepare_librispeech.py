import os
import glob
import argparse
import unicodedata
from multiprocessing import cpu_count

from tqdm.auto import tqdm
import sentencepiece as sp
from datasets.dataset_base import BUFFER_SIZE

import utils
from datasets.librispeech_dataset import LibrispeechDataset

UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 1
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 2
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 3
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 0  # unused, by default

def get_transcripts(data_dir, data_types):
    """ Collect all transcripts corresponding to data types """
    txt_list = []
    for data_type in data_types:
        for speaker_id in os.listdir(os.path.join(data_dir, data_type)):
            if speaker_id == '.DS_Store': 
                continue

            for chapter_id in os.listdir(os.path.join(data_dir, data_type, speaker_id)):
                if chapter_id == '.DS_Store': 
                    continue

                file_name = os.path.join(data_dir, data_type, speaker_id, chapter_id, speaker_id + '-' + chapter_id + '.trans.txt')
                txt_list.append(file_name)

    transcripts = []
    for txt_file_name in txt_list:
        with open(txt_file_name, "r") as f:
            for line in f.readlines():
                x = line.split('\n')[0].split(' ')
                uttid = x[0]
                transcript = ' '.join(x[1:])
                spkid, chapter_id, _ = uttid.split('-')
                wav_file_name = os.path.join(data_dir, data_type, spkid, chapter_id, uttid + '.flac')
                if not os.path.exists(wav_file_name):
                    continue
                transcripts.append(transcript)
    return transcripts




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = utils.load_yaml(args.config)

    train_data_types = ["train-clean-100", "train-clean-360"]
    valid_data_types = ["dev-clean"]

    # create transcripts
    transcripts = get_transcripts(args.data_dir, train_data_types)

    # train sentence-piece model & save it
    output_prefix = os.path.join(args.out_dir, "sentencepiece")

    def sentence_iterator():
        for line in transcripts:
            yield line

    # train sentence-piece model
    sp.SentencePieceTrainer.Train(
        sentence_iterator=sentence_iterator(),
        model_prefix=output_prefix,
        model_type=config["text_config"]["model_type"],
        vocab_size=config["text_config"]["target_vocab_size"],
        num_threads=cpu_count(),
        unk_id=UNK_TOKEN_ID,
        bos_id=BOS_TOKEN_ID,
        eos_id=EOS_TOKEN_ID,
        pad_id=PAD_TOKEN_ID,
        unk_surface="__UNKNOWN__",  # change default unk surface U+2047("⁇") by "__UNKNOWN__"
    )
        
    # create tf-record for training
    libri_train = LibrispeechDataset(
        output_prefix + ".model",
        config=config["dsp_config"]
    )
    libri_train.create(args.data_dir, train_data_types)
    train_tfrecord_name = os.path.join(args.out_dir, "librispeech", "train.tfrecord")
    libri_train.save(train_tfrecord_name)

    # create tf-record for validation
    libri_valid = LibrispeechDataset(
        output_prefix + ".model",
        config=config["dsp_config"]
    )
    libri_valid.create(args.data_dir,valid_data_types)
    valid_tfrecord_name = os.path.join(args.out_dir, "librispeech", "valid.tfrecord")
    libri_valid.save(valid_tfrecord_name)
