import os
import argparse
import unicodedata
from tqdm.auto import tqdm
from typing import List


def get_librispeech_text_files(data_dir, data_types):
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
    return txt_list

def sort_entry(entry: List[type], sampling_rate: int = 16000) -> List[type]:
    def get_file_size(file) -> int:
        return os.stat(file).st_size
        
    sorted_entry = sorted(entry, key=lambda x : get_file_size(x[0]))
    return sorted_entry
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Setup LibriSpeech Transcripts")
    parser.add_argument("--datadir", type=str, required=True, help="Directory of dataset")
    parser.add_argument("--outdir", type=str, required=True, help="The output .tsv transcript file path")
    args = parser.parse_args()

    train_data_types = ["train-clean-100", "train-clean-360"]
    valid_data_types = ["dev-clean"]

    
    # Train
    train_entry = []
    train_text_files = get_librispeech_text_files(args.datadir, train_data_types)
    for text_file in tqdm(train_text_files, desc="[Loading Train]"):
        current_dir = os.path.dirname(text_file)
        with open(text_file, "r", encoding="utf-8") as txt:
            lines = txt.read().splitlines()
        for line in lines:
            line = line.split(" ", maxsplit=1)
            audio_file = os.path.join(current_dir, line[0] + ".flac")
            text = unicodedata.normalize("NFC", line[1].lower())
            #train_entry.append(f"{audio_file}\t{text}\n")
            train_entry.append([audio_file, text])
            
    # Sort entry with duration
    train_entry = sort_entry(train_entry)

    train_tsv = os.path.join(args.outdir, "train_entry.tsv")
    with open(train_tsv, "w", encoding="utf-8") as out:
        out.write("PATH\tTRANSCRIPT\n")
        for (audio_file, text) in tqdm(train_entry, desc="[Writing Train]"):
            out.write(f"{audio_file}\t{text}\n")

    # Valid
    valid_entry = []
    valid_text_files = get_librispeech_text_files(args.datadir, valid_data_types)
    for text_file in tqdm(valid_text_files, desc="[Loading Valid]"):
        current_dir = os.path.dirname(text_file)
        with open(text_file, "r", encoding="utf-8") as txt:
            lines = txt.read().splitlines()
        for line in lines:
            line = line.split(" ", maxsplit=1)
            audio_file = os.path.join(current_dir, line[0] + ".flac")
            text = unicodedata.normalize("NFC", line[1].lower())
            #valid_entry.append(f"{audio_file}\t{text}\n")
            valid_entry.append([audio_file, text])

    # Sort entry with duration
    valid_entry = sort_entry(valid_entry)
    
    valid_tsv = os.path.join(args.outdir, "valid_entry.tsv")
    with open(valid_tsv, "w", encoding="utf-8") as out:
        out.write("PATH\tTRANSCRIPT\n")
        for (audio_file, text) in tqdm(valid_entry, desc="[Writing valid]"):
            out.write(f"{audio_file}\t{text}\n")


    