# Realtime DDSP in PyTorch + Export to neutone

## Features

- Basic DDSP modules
    - Harmonic Synthesis
    - Filtered Noise Synthesizer
    - Convolution Reverb
- Basic estimator network for DDSP
    - F0 and loudness input
- [Neutone](https://github.com/QosmoInc/neutone_sdk) compatibility
    - Replaces DDSP modules with streaming compatible modules

## Usage

First, install the required packages.
```
pip install -r requirements.txt
```

Then, run the training script.
```
python train.py name=NAME_OF_OUTPUT data.raw_dir=PATH/TO/.WAV/DIRECTORY data.db_path=PATH/TO/TEMPORARY/DATABASE/DIRECTORY
```

The results including checkpoint files (.ckpt) and tensorboard logs are saved under `logs/{name}`.
Then, you can export the checkpoint file to neutone model (.nm) file.

```
python export.py CKPT_FILE EXPORT_NAME
```
The model.nm file along with reconstruction samples will be outputted to `exports/EXPORT_NAME`. You can load these files in the neutone plugin. Make sure to fill in the model details in `DDSPModelWrapper` when submitting these models to neutone.

### Arguments

This project uses [hydra](https://hydra.cc/) to manage configurations. Basically, the config files are split into multiple files under `configs/`. For example, the default data configuration is specified in `configs/model/slice.yaml`. You can then override these settings by changing `config/config.yaml`, and can also be overwritten in command line like `data.raw_dir=PATH/TO/.WAV/DIRECTORY`.

Here are some extra arguments you might want to edit:
- sample_rate (default: 48000)
- ckpt (default: None)
    - Specify a checkpoint to resume training from.
- batch_size (default: [32])
- data.f0_range (default: [32, 2000])
    - The range of the pitch present in the training data in Hz.
    - Narrowing down this value to the range of the instrument can help detect pitch more accurately.
- data.f0_viterbi (default: true)
    - Viterbi algorithm is used for smoothing out pitch and removing jumps.
    - Instruments with a lot of pitch slides might be better without it.
- trainer.max_steps (default: 100000)
    - Ends training after `trainer.max_steps` steps.
- ckpt_nsteps (default: 10000)
    - .ckpt files will be saved every `ckpt_nsteps` steps.
- model.lr (default: 0.0001)
    - Model learning rate
- model.estimator.hidden_size (default: 512)
    - Size of the estimator network

### Tips

- Train with monophonic melodies!
    - Pitch detection only works for monophonic sounds.
    - DDSP assumes that there is a single pitch to the audio.
- Train with a performance of a single instrument!
    - DDSP works best for modeling a single instrument.
    - Current estimator only uses pitch and loudness as input, which doesn't allow for diverse timbre.

## Details

### Preprocessing (`data.py`)

It will first check if the database file has already been created under `data.db_path`. If a database file already exists, it will use that database file. If not, it will load the wave files in the directory specified by `data.raw_dir`, and perform preprocessing. The audio file is cut into 1 second segments and its pitch is detected using [torchcrepe](https://github.com/maxrmorrison/torchcrepe). The preprocessing results including the sliced audio and pitch is saved into a database file under `data.db_path`.

### Model (`model.py`, `estimator.py`)

The model is the same as a basic DDSP model. An estimator network is trained to predict the parameters for the DDSP synthesizer. The DDSP synthesizer outputs audio from the estimated parameters. The outputs of the DDSP synthesizer is used to calculate the multi-scale spectrogram loss from the original input. For details about the DDSP model, see Google Magenta's [blog post](https://magenta.tensorflow.org/ddsp).

### DDSP Synthesizer (`synthesizer.py`, `processor.py`, `modules/`)

The synthesizer architecture can be flexibly constructed as a yaml file. For example, the default synthesizer config `configs/synth/hpnir.yaml` instantiates Harmonic (harmonic synthesizer), FilteredNoise (noise synthesizer), and IRReverb (convolution reverb) modules and specifies the connections for each modules.

The reverb module (`reverb.py`, `IRReverb`) is a convolution reverb. This IR is fixed over the entire dataset and is learned as a model parameter. 

### Streaming (`stream.py`, `export.py`)

The original DDSP modules are not compatible for streaming synthesis, as they rely on future information for interpolation, etc. `export.py` converts the model into a streaming compatible model using caches. For pitch detection, since the CREPE model only supports a single sample rate (16kHz), we instead use YIN.

## Credits

This project is based on the original [DDSP paper by Engel et al](https://magenta.tensorflow.org/ddsp).
This project also uses the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) (MIT License).