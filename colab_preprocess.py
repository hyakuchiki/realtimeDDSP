import os, argparse, warnings
from diffsynth.data import SliceDataset
import pytorch_lightning as pl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', type=str)
    parser.add_argument('db_path', type=str)
    parser.add_argument('--sr', type=int)
    parser.add_argument('--length', type=float, default=1.0)
    parser.add_argument('--frame_rate', type=int, default=50)
    parser.add_argument('--f0_range', type=int, nargs=2)
    parser.add_argument('--f0_viterbi', type=bool)
    args = parser.parse_args()

    pl.seed_everything(seed=0, workers=True)
    warnings.simplefilter("once")
    # load dataset
    print('Starting Preprocessing.')
    dataset = SliceDataset(args.raw_dir, args.db_path, args.sr, args.length, args.frame_rate, args.f0_range, args.f0_viterbi)
    print(f'Loaded {len(dataset)} samples.')