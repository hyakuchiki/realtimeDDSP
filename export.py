from diffsynth.stream import StreamEstimatorFLSynth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt',             type=str,   help='')
    parser.add_argument('--seed',     type=int,   default=0, help='')
    parser.add_argument('--batch_size',     type=int,   default=64, help='')
    parser.add_argument('--write_audio',    action='store_true')
    parser.add_argument('--ext_f0',    action='store_true')
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = 'cuda'