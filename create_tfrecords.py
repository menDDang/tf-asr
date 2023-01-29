import argparse
import os

from rich import traceback

import tfasr

traceback.install()
LOG = tfasr.utils.logging.DetailLogger(__name__, multi=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nthreads', 
        type=int, 
        default=4, 
        help='number of thread to be used parallel jobs')
    parser.add_argument(
        '--outdir', 
        type=str, 
        required=True)
    args = parser.parse_args()
    
    # Force to use CPU
    tfasr.utils.set_devices([], cpu=False)

    # Parse runtime arguments
    outdir = args.outdir
    
    # Get entry files
    train_entry_path = os.path.join(outdir, 'train_entry.tsv')
    with open(train_entry_path, 'r') as f:
        train_entries = f.readlines()
    train_entries = train_entries[1:] # erase first line
    train_entries = [line.split('\t') for line in train_entries]

    valid_entry_path = os.path.join(outdir, 'valid_entry.tsv')
    with open(valid_entry_path, 'r') as f:
        valid_entry = f.readlines()
    valid_entry = valid_entry[1:]
    valid_entry = [line.split('\t') for line in valid_entry]


    tfasr.data.datasets.ASRDataSet.create(
        total_entry=train_entries,
        out_dir=os.path.join(outdir, 'train'),
        num_processes=args.nthreads)
        
    tfasr.data.datasets.ASRDataSet.create(
        total_entry=valid_entry,
        out_dir=os.path.join(outdir, 'valid'),
        num_processes=args.nthreads)