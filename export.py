import os, logging, argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor

from diffsynth.model import EstimatorSynth
from diffsynth.modules.generators import FilteredNoise, Harmonic
from diffsynth.modules.reverb import IRReverb
from diffsynth.synthesizer import Synthesizer
from diffsynth.processor import Mix
from diffsynth.stream import StreamFilteredNoise, StreamIRReverb, StreamHarmonic

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

def get_stream_synth(synth):
    new_ps = []
    new_cs = []
    conditioned = synth.conditioned_params
    for proc, conn in zip(synth.processors, synth.connections):
        if isinstance(proc, Harmonic):
            # Replace with streamable version of harmonic synthesizer
            new_ps.append(StreamHarmonic(proc.sample_rate, proc.normalize_below_nyquist, proc.name, proc.n_harmonics, proc.freq_range))
            new_cs.append(conn)
        elif isinstance(proc, FilteredNoise):
            # Replace with streamable version of noise synthesizer
            new_ps.append(StreamFilteredNoise(proc.filter_size, proc.name, proc.amplitude))
            new_cs.append(conn)
        elif isinstance(proc, IRReverb):
            # Replace with streamable version of ir reverb
            new_ps.append(StreamIRReverb(proc.ir, proc.name))
            conn_mix = dict(conn)
            # this version has a parameter for adjusting reverb mix
            conn_mix['mix'] = 'irmix'
            new_cs.append(conn_mix)
            conditioned.append('irmix')
        elif proc.name == 'add':
            # Replace add module with mix module for adjusting harm/noise
            new_ps.append(Mix(proc.name))
            conn_mix = dict(conn)
            conn_mix['mix_a'] = 'harmmix'
            conn_mix['mix_b'] = 'noisemix'
            new_cs.append(conn_mix)
            conditioned.extend(['harmmix', 'noisemix'])
        else:
            new_ps.append(proc)
            new_cs.append(conn)
    synth.processors = torch.nn.ModuleList(new_ps)
    # make new synth
    dag = [(proc, conn) for proc, conn in zip(new_ps, new_cs)]
    new_synth = Synthesizer(dag, conditioned=conditioned)
    return new_synth

class DDSPModelWrapper(WaveformToWaveformBase):
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
        return [
            NeutoneParameter(name='Pitch Shift', description='Magnitude of latent noise', default_value=0.5),
            NeutoneParameter(name='Harmonics Mix', description='Mix of harmonic synthesizer', default_value=0.5),
            NeutoneParameter(name='Noise Mix', description='Mix of noise synthesizer', default_value=0.5),
            NeutoneParameter(name='Reverb Mix', description='Mix of IR reverb', default_value=0.5),
            ]
    def is_input_mono(self) -> bool:
        return True

    def is_output_mono(self) -> bool:
        return True

    def get_native_sample_rates(self) -> List[int]:
        return [48000]

    def get_native_buffer_sizes(self) -> List[int]:
        return [2048]

    def get_citation(self) -> str:
        return """
        Engel, J., Hantrakul, L., Gu, C., & Roberts, A. (2020). DDSP:Differentiable Digital Signal Processing. ICLR.
        """

    @torch.no_grad()
    def do_forward_pass(
        self, x: Tensor,
        params: Dict[str, torch.Tensor]
    ) -> Tensor:
        if x.size(0) == 2:
            x = x.mean(dim=0, keepdim=True)
        # pitch shift parameter
        MAX_SHIFT = 24 # semitones
        pshift = (params['Pitch Shift'] - 0.5) * 2 * MAX_SHIFT # -24~24
        semishift = torch.round(pshift)
        f0_mult = 2**(semishift/12)
        # Harmonics/Noise, reverb mix
        harm_mix = params['Harmonics Mix'] * 2 # 0(no harmonics)~2
        noise_mix = params['Noise Mix'] * 2 # 0(no noise)~2
        rev_mix = params['Reverb Mix'] # 0(no reverb)~1(reverb only)
        cond_params = {'harmmix': harm_mix, 'noisemix': noise_mix, 'irmix': rev_mix}
        out, data = self.model(x, f0_mult=f0_mult, param=cond_params)
        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt',             type=str,   help='')
    parser.add_argument('output',           type=str,   help='model output name')
    parser.add_argument('--folder',   default='./exports', help='output folder')
    parser.add_argument('--sounds',   nargs='*', type=str, default=None, help='directory of sounds to use as example input.')
    args = parser.parse_args()
    root_dir = Path(args.folder) / args.output

    model = EstimatorSynth.load_from_checkpoint(args.ckpt)
    replace_modules(model.estimator)
    # get streamable hpnir synth with mix parameters
    model.synth = get_stream_synth(model.synth)
    stream_model = CachedStreamEstimatorFLSynth(model.estimator, model.synth, 48000, hop_size=960)
    dummy = torch.zeros(1, 2048)
    _ = stream_model(dummy, torch.ones(1), {'harmmix': torch.ones(1), 'noisemix': torch.ones(1), 'irmix': torch.ones(1)*0.5})
    wrapper = DDSPModelWrapper(stream_model)

    soundpairs = None
    if args.sounds is not None:
        soundpairs = []
        for sound in args.sounds:
            wave, sr = torchaudio.load(sound)
            input_sample = AudioSample(wave, sr)
            rendered_sample = render_audio_sample(wrapper, input_sample)
            soundpairs.append(AudioSamplePair(input_sample, rendered_sample))

    save_neutone_model(
        wrapper, root_dir, freeze=False, dump_samples=True, submission=True, audio_sample_pairs=soundpairs
    )