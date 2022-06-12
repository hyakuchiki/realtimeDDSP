import json, os, logging, argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor

from diffsynth.model import EstimatorSynth
from diffsynth.stream import CachedStreamEstimatorFLSynth, replace_modules
from neutone_sdk.audio import (
    AudioSample,
    AudioSamplePair,
    render_audio_sample,
)
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model
import torchaudio

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

class RAVEModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "DDSP.example"

    def get_model_authors(self) -> List[str]:
        return ["Author Name"]

    def get_model_short_description(self) -> str:
        return "DDSP model trained on some data"

    def get_model_long_description(self) -> str:
        return "DDSP timbre transfer model explanation. Useful for ~ sounds."

    def get_technical_description(self) -> str:
        return "DDSP model proposed by Jesse Engel et al."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://openreview.net/pdf?id=B1x1ma4tDr",
            "Code": "https://github.com/magenta/ddsp",
        }

    def get_tags(self) -> List[str]:
        return ["timbre transfer", "DDSP"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        """
        set to True for models in experimental stage
        (status shown on the website)
        """
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return []

    def is_input_mono(self) -> bool:
        return True

    def is_output_mono(self) -> bool:
        return True

    def get_native_sample_rates(self) -> List[int]:
        return [48000]

    def get_native_buffer_sizes(self) -> List[int]:
        return [2048]

    def get_citation(self) -> str:
        return """Caillon, A., & Esling, P. (2021). RAVE: A variational autoencoder for fast and high-quality neural audio synthesis. arXiv preprint arXiv:2111.05011."""

    @torch.no_grad()
    def do_forward_pass(
        self, x: Tensor,
    ) -> Tensor:
        # Currently VST input-output is mono, which matches RAVE.
        if x.size(0) == 2:
            x = x.mean(dim=0, keepdim=True)
        out = self.model(x)
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt',             type=str,   help='')
    parser.add_argument('output',           type=str,   help='model output name')
    parser.add_argument('--folder',   default='./exports', help='output folder')
    parser.add_argument('--sounds',   nargs='*', type=str, default=None, help='directory of sounds to use as example input.')
    args = parser.parse_args()
    root_dir = Path(args.folder) / args.output

    # wrap it
    model = torch.jit.load(args.input)
    wrapper = RAVEModelWrapper(model)

    soundpairs = None
    if args.sounds is not None:
        soundpairs = []
        for sound in args.sounds:
            wave, sr = torchaudio.load(sound)
            input_sample = AudioSample(wave, sr)
            rendered_sample = render_audio_sample(wrapper, input_sample)
            soundpairs.append(AudioSamplePair(input_sample, rendered_sample))

    model = EstimatorSynth.load_from_checkpoint(args.ckpt)
    replace_modules(model.estimator)
    stream_model = CachedStreamEstimatorFLSynth(model.estimator, model.synth_cfg, 48000, hop_size=960)

    save_neutone_model(
        wrapper, root_dir, freeze=False, dump_samples=True, submission=True, audio_sample_pairs=soundpairs
    )