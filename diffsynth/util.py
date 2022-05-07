import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class StatsLog():
    def __init__(self):
        self.stats = dict()

    def __getitem__(self, key):
        return sum(self.stats[key]) / len(self.stats[key])

    def average(self):
        return {k: sum(v)/len(v) for k, v in self.stats.items()}

    def std(self):
        return {k: np.array(v).std() for k, v in self.stats.items()}

    def add_entry(self, k, v):
        if isinstance(v, torch.Tensor):
            # turn into list of python floats
            v = v.squeeze().tolist()
        if not isinstance(v, list):
            # a lone float into list of float
            v = [v]
        if k in self.stats:
            self.stats[k].extend(v)
        else:
            self.stats[k] = v

    def update(self, stat_dict):
        for k, v in stat_dict.items():
            self.add_entry(k, v)

def log_eps(x, eps=1e-4):
    return torch.log(x+eps)

def exp_scale(x, log_exponent=3, max_value=2.0, threshold=1e-7):
    return max_value * x**log_exponent + threshold

def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    """Exponentiated Sigmoid pointwise nonlinearity.

    Bounds input to [threshold, max_value] with slope given by exponent.

    Args:
    x: Input tensor.
    exponent: In nonlinear regime (away from x=0), the output varies by this
    factor for every change of x by 1.0.
    max_value: Limiting value at x=inf.
    threshold: Limiting value at x=-inf. Stablizes training when outputs are
        pushed to 0.

    Returns:
        A tensor with pointwise nonlinearity applied.
    """
    return max_value * torch.sigmoid(x)**math.log(exponent) + threshold

def sin_synthesis(frequencies, amplitudes, n_samples = 64000, sample_rate = 16000, fm_signal=None):
    """wavetable synthesis similar to the one in DDSP

    Args:
        frequencies (torch.Tensor): Frame-wise frequency in Hz [batch_size, n_frames, 1]
        amplitudes (torch.Tensor): Frame-wise amplitude envelope [batch_size, n_frames, 1]
        fm_signal: audio rate signal for FM (phase modulation)
    Returns:
        audio: Sin at the frequency and amplitude of the inputs. Shape[batch_size, n_samples]
    """
    # upsample frequency/amplitude envelope
    batch_size = frequencies.shape[0]

    if len(frequencies.shape) == 3:
        frequencies = frequencies.squeeze(-1) #[batch_size, n_frames]
    if len(amplitudes.shape) == 3:
        amplitudes =  amplitudes.squeeze(-1) #[batch_size, n_frames]

    frq_env = resample_frames(frequencies, n_samples) # [batch_size, n_samples]
    amp_env = resample_frames(amplitudes, n_samples) # TODO should really use windows to avoid jaggy envelope
    
    phase_velocity = frq_env / float(sample_rate)
    phase = torch.cumsum(phase_velocity, 1)[:, :-1] % 1.0 # [batch_size, n_samples]
    phase = torch.cat([torch.zeros(batch_size, 1).to(phase.device), phase], dim=1) # exclusive cumsum starting at 0
    phase_rad = phase * 2 * np.pi
    if fm_signal is not None:
        audio = torch.sin(phase_rad+fm_signal)
    else:
        audio = torch.sin(phase_rad)
    audio *= amp_env
    return audio

def wavetable_synthesis(frequencies, amplitudes, wavetable, n_samples = 64000, sample_rate = 16000, fm_signal=None):
    """wavetable synthesis similar to the one in DDSP

    Args:
        frequencies (torch.Tensor): Frame-wise frequency in Hz [batch_size, n_frames, 1]
        amplitudes (torch.Tensor): Frame-wise amplitude envelope [batch_size, n_frames, 1]
        wavetable: oscillator waveform can change each frame ([batch_size, n_frames, len_waveform]) or be constant ([batch_size, len_waveform])
        fm_signal: audio rate signal for FM (phase modulation)
    Returns:
        audio: Audio at the frequency and amplitude of the inputs, with harmonics given by the wavetable. Shape[batch_size, n_samples]
    """
    # upsample frequency/amplitude envelope
    input_shape = frequencies.shape
    batch_size = input_shape[0]

    if len(frequencies.shape) == 3:
        frequencies = frequencies.squeeze(-1) #[batch_size, n_frames]
    if len(amplitudes.shape) == 3:
        amplitudes =  amplitudes.squeeze(-1) #[batch_size, n_frames]

    frq_env = resample_frames(frequencies, n_samples) # [batch_size, n_samples]
    amp_env = resample_frames(amplitudes, n_samples) # TODO should really use windows to avoid jaggy envelope

    # interpolate wavetable
    if len(wavetable.shape) == 3 and wavetable.shape[1] > 1:
        wavetable = resample_frames(wavetable, n_samples)
    
    phase_velocity = frq_env / float(sample_rate)
    phase = torch.cumsum(phase_velocity, 1)[:, :-1] % 1.0 # [batch_size, n_samples]
    phase = torch.cat([torch.zeros(batch_size, 1).to(phase.device), phase], dim=1) # exclusive cumsum starting at 0
    if fm_signal is not None:
        audio = linear_lookup(phase+fm_signal / (2*np.pi) , wavetable)
    else:
        audio = linear_lookup(phase, wavetable)
    audio *= amp_env
    return audio

def linear_lookup(phase, wavetable):
    """Lookup from wavetables

    Args:
        phase: instantaneous phase of base oscillator (0.0~1.0) [batch_size, n_samples]
        wavetable ([type]): [batch_size, n_samples, len_waveform] or [batch_size, len_waveform]
    """
    phase = phase[:, :, None]
    len_waveform = wavetable.shape[-1]
    phase_wavetable = torch.linspace(0.0, 1.0, len_waveform).to(wavetable.device)
    if len(wavetable.shape) == 2:
        wavetable = wavetable.unsqueeze(1)

    # Get pair-wise distances from the oscillator phase to each wavetable point.
    # Axes are [batch, time, len_waveform]. NOTE: <- this is super large
    phase_distance = abs((phase - phase_wavetable[None, None, :]))
    phase_distance *= len_waveform - 1
    # Weighting for interpolation.
    # Distance is > 1.0 (and thus weights are 0.0) for all but nearest neighbors.
    weights = nn.functional.relu(1.0 - phase_distance) # [batch_size, n_samples, len_waveform]
    weighted_wavetables = weights * wavetable
    return torch.sum(weighted_wavetables, dim=-1)
    
def resample_frames(inputs, n_timesteps, mode='linear', add_endpoint=True):
    """interpolate signals with a value each frame into signal with a value each timestep
    [n_frames] -> [n_timesteps]

    Args:
        inputs (torch.Tensor): [n_frames], [batch_size, n_frames], [batch_size, n_frames, channels]
        n_timesteps (int): 
        mode (str): 'window' for interpolating with overlapping windows
        add_endpoint ([type]): I think its for windowed interpolation
    Returns:
        torch.Tensor
        [n_timesteps], [batch_size, n_timesteps], or [batch_size, n_timesteps, channels?]
    """
    orig_shape = inputs.shape

    if len(orig_shape)==1:
        inputs = inputs.unsqueeze(0) # [dummy_batch, n_frames]
        inputs = inputs.unsqueeze(1) # [dummy_batch, dummy_channel, n_frames]
    if len(orig_shape)==2:
        inputs = inputs.unsqueeze(1) # [batch, dummy_channel, n_frames]
    if len(orig_shape)==3:
        inputs = inputs.permute(0, 2, 1) # # [batch, channels, n_frames]

    if mode == 'window':
        raise NotImplementedError
        # upsample_with_windows(outputs, n_timesteps, add_endpoint)
    else:
        # interpolate expects [batch_size, channel, (depth, height,) width]
        outputs = nn.functional.interpolate(inputs, size=n_timesteps, mode=mode, align_corners=not add_endpoint)
    
    if len(orig_shape) == 1:
        outputs = outputs.squeeze(1) # get rid of dummy channel 
        outputs = outputs.squeeze(0) #[n_timesteps]
    if len(orig_shape) == 2:
        outputs = outputs.squeeze(1) # get rid of dummy channel # [n_frames, n_timesteps] 
    if len(orig_shape)==3:
        outputs = outputs.permute(0, 2, 1) # [batch, n_frames, channels]

    return outputs

def get_harmonic_frequencies(frequencies, n_harmonics):
    """Create integer multiples of the fundamental frequency.

    Args:
    frequencies: Fundamental frequencies (Hz). Shape [batch_size, time, 1].
    n_harmonics: Number of harmonics.

    Returns:
    harmonic_frequencies: Oscillator frequencies (Hz).
        Shape [batch_size, time, n_harmonics].
    """
    f_ratios = torch.linspace(1.0, float(n_harmonics), int(n_harmonics)).to(frequencies.device)
    f_ratios = f_ratios[None, None, :]
    harmonic_frequencies = frequencies * f_ratios
    return harmonic_frequencies

def remove_above_nyquist(frequency_envelopes, amplitude_envelopes, sample_rate=16000):
    """Set amplitudes for oscillators above nyquist to 0.

    Args:
        frequency_envelopes: Sample/frame-wise oscillator frequencies (Hz). Shape
        [batch_size, n_samples(or n_frames), n_sinusoids].
        amplitude_envelopes: Sample/frame-wise oscillator amplitude. Shape [batch_size,
        n_samples(or n_frames), n_sinusoids].
        sample_rate: Sample rate in samples per a second.

    Returns:
        amplitude_envelopes: Sample-wise filtered oscillator amplitude.
        Shape [batch_size, n_samples, n_sinusoids].
    """
    if amplitude_envelopes.shape[1] != frequency_envelopes.shape[1]:
        frequency_envelopes = resample_frames(frequency_envelopes, amplitude_envelopes.shape[1])

    amplitude_envelopes = torch.where(torch.ge(frequency_envelopes, sample_rate / 2.0), torch.zeros_like(amplitude_envelopes), amplitude_envelopes)
    return amplitude_envelopes

def harmonic_synthesis(frequencies, amplitudes, harmonic_shifts=None, harmonic_distribution=None, n_samples=64000, sample_rate=16000, amp_resample_method='window'):
    """Generate audio from frame-wise monophonic harmonic oscillator bank.

    Args:
        frequencies: Frame-wise fundamental frequency in Hz. Shape [batch_size, n_frame, 1].
        amplitudes: Frame-wise oscillator peak amplitude. Shape [batch_size, n_frames, 1].
        harmonic_shifts: Harmonic frequency variations (Hz), zero-centered. Total frequency of a harmonic is equal to (frequencies * harmonic_number * (1 + harmonic_shifts)). Shape [batch_size, n_frames, n_harmonics].
        harmonic_distribution: Harmonic amplitude variations, ranged zero to one. Total amplitude of a harmonic is equal to (amplitudes * harmonic_distribution). Shape [batch_size, n_frames, n_harmonics].
        n_samples: Total length of output audio. Interpolates and crops to this.
        sample_rate: Sample rate.
        amp_resample_method: Mode with which to resample amplitude envelopes.

    Returns:
        audio: Output audio. Shape [batch_size, n_samples]
    """

    if harmonic_distribution is not None:
        n_harmonics = harmonic_distribution.shape[-1]
    elif harmonic_shifts is not None:
        n_harmonics = harmonic_shifts.shape[-1]
    else:
        n_harmonics = 1

    # Create harmonic frequencies [batch_size, n_frames, n_harmonics].
    harmonic_frequencies = get_harmonic_frequencies(frequencies, n_harmonics)
    if harmonic_shifts is not None:
        harmonic_frequencies *= (1.0 + harmonic_shifts)

    # Create harmonic amplitudes [batch_size, n_frames, n_harmonics].
    if harmonic_distribution is not None:
        harmonic_amplitudes = amplitudes * harmonic_distribution
    else:
        harmonic_amplitudes = amplitudes

    # Create sample-wise envelopes.
    frequency_envelopes = resample_frames(harmonic_frequencies, n_samples)  # cycles/sec
    # amplitude_envelopes = resample_frames(harmonic_amplitudes, n_samples,method=amp_resample_method)
    amplitude_envelopes = resample_frames(harmonic_amplitudes, n_samples) # window has not been implemented yet
    # Synthesize from harmonics [batch_size, n_samples].
    audio = oscillator_bank(frequency_envelopes,
                            amplitude_envelopes,
                            sample_rate=sample_rate)
    return audio

def oscillator_bank(frequency_envelopes, amplitude_envelopes, sample_rate=16000, sum_sinusoids=True):
    """Generates audio from sample-wise frequencies for a bank of oscillators.

    Args:
        frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape [batch_size, n_samples, n_sinusoids].
        amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size, n_samples, n_sinusoids].
        sample_rate: Sample rate in samples per a second.
        sum_sinusoids: Add up audio from all the sinusoids.

    Returns:
        wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids] if sum_sinusoids=False, else shape is [batch_size, n_samples].
    """

    # Don't exceed Nyquist.
    amplitude_envelopes = remove_above_nyquist(frequency_envelopes, amplitude_envelopes, sample_rate)

    # Change Hz to radians per sample.
    omegas = frequency_envelopes * (2.0 * np.pi)  # rad / sec
    omegas = omegas / float(sample_rate)  # rad / sample

    # Accumulate phase and synthesize.
    output = torch.cumsum(omegas, 1)
    output = torch.sin(output)
    output = amplitude_envelopes * output  # [batch_size, n_samples, n_sinusoids]
    if sum_sinusoids:
        output = torch.sum(output, dim=-1)  # [batch_size, n_samples]
    return output

def get_fft_size(frame_size: int, ir_size: int) -> int:
    """Calculate final size for efficient FFT. power of 2
    fft size should be greater than frame_size + ir_size
    Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.

    Returns:
    fft_size: Size for efficient FFT.
    """
    convolved_frame_size = ir_size + frame_size - 1
    # Next power of 2.
    return int(2**np.ceil(np.log2(convolved_frame_size)))

def frame_signal(signal, frame_size):
    """
    cut signal into nonoverlapping frames
    Args:
        signal: [batch, n_samples]
        frame_size: int 
    Returns:
        [batch, n_frames, frame_size]
    """
    signal_len = signal.shape[-1]
    padding = (frame_size - (signal_len % frame_size) ) % frame_size
    signal = torch.nn.functional.pad(signal, (0, padding), 'constant', 0)
    frames = torch.split(signal.unsqueeze(1), frame_size, dim=-1)
    return torch.cat(frames, dim=1)

def slice_windows(signal, frame_size, hop_size, window=None):
    """
    slice signal into overlapping frames
    pads end if (l_x - frame_size) % hop_size != 0
    Args:
        signal: [batch, n_samples]
        frame_size (int): size of frames
        hop_size (int): size between frames
    Returns:
        [batch, n_frames, frame_size]
    """
    _batch_dim, l_x = signal.shape
    remainder = (l_x - frame_size) % hop_size
    pad = 0 if (remainder == 0) else hop_size - remainder
    signal = F.pad(signal, (0, pad), 'constant')
    signal = signal[:, None, None, :] # adding dummy channel/height
    frames = F.unfold(signal, (1, frame_size), stride=(1, hop_size)) #batch, frame_size, n_frames
    frames = frames.permute(0, 2, 1) # batch, n_frames, frame_size
    if window == 'hamming':
        win = torch.hamming_window(frame_size)[None, None, :].to(frames.device)
        frames = frames * win
    return frames

def variable_delay(phase, audio, buf_size):
    """delay with variable length

    Args:
        phase (torch.Tensor): 0~1 0: no delay 1: delay=max_length (batch, n_samples)
        audio (torch.Tensor): audio signal (batch, n_samples)
        buf_size (int)    : buffer size in samples = max delay length

    Returns:
        torch.Tensor: delayed audio (batch, n_samples)
    """
    batch_size, n_samples = audio.shape
    audio_4d = audio[:, None, None, :] # (B, C=1, H=1, W=n_samples)
    delay_ratio = buf_size*2/n_samples
    grid_x = torch.linspace(-1, 1, n_samples, device=audio.device)[None, :]
    grid_x = grid_x - delay_ratio + delay_ratio*phase # B, W=n_samples
    grid_x = grid_x[:, None, :, None] # B, H=1, W=n_samples, 1
    grid_y = torch.zeros(batch_size, 1, n_samples, 1, device=audio.device) # # B, H=1, W=n_samples, 1
    grid = torch.cat([grid_x, grid_y], dim=-1)
    output = torch.nn.functional.grid_sample(audio_4d, grid, align_corners=True)
    # shape: (B, C=1, H=1, W)
    output = output.squeeze(2).squeeze(1)
    return output

    # pad_sig = torch.nn.functional.pad(audio, (buf_size-1, 0))
    # assert orig_len % proc_len == 0
    # if proc_len is None:
    #     frames = slice_windows(pad_sig, buf_size, 1) # (b, n_samples, buf_size)
    #     return linear_lookup(phase, frames)
    # else:
    #     output = []
    #     sig_frames = slice_windows(pad_sig, proc_len+buf_size-1, proc_len)
    #     n_frames = sig_frames.shape[1]
    #     for i in range(n_frames):
    #         sig = sig_frames[:, i] # (batch, proc_len+buf_size-1)
    #         ph = phase[:, i*proc_len:(i+1)*proc_len]
    #         ll_frames = slice_windows(sig, buf_size, 1) # (batch, proc_len, max_length)
    #         output.append(linear_lookup(ph, ll_frames)) # (batch, proc_len)
    #     output = torch.cat(output, dim=-1)
    #     output = output[:, :orig_len]
    #     return output

def overlap_and_add(signal, frame_step):
    """overlap-add signals ported from tf.signals

    Args:
        signal (torch.Tensor): (batch_size, frames, frame_length)
        frame_step (int): size of overlap offset
    Returns:
    A `Tensor` with shape `[..., output_size]`
    """
    batch_size = signal.shape[0]
    n_frames = signal.shape[1]
    frame_length = signal.shape[2]

    output_length = frame_length + frame_step * (n_frames - 1)
    if frame_length == frame_step:
        return signal.flatten(-2, -1)
    segments = -(-frame_length // frame_step)
    pad_width = segments * frame_step - frame_length
    # The following code is documented using this example:
    #
    # frame_step = 2
    # signal.shape = (3, 5)
    # a b c d e
    # f g h i j
    # k l m n o

    # Pad the frame_length dimension to a multiple of the frame step.
    # Pad the frames dimension by `segments` so that signal.shape = (6, 6)
    # a b c d e 0
    # f g h i j 0
    # k l m n o 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    signal = nn.functional.pad(signal, (0, pad_width, 0, segments))
    # Reshape so that signal.shape = (6, 3, 2)
    # ab cd e0
    # fg hi j0
    # kl mn o0
    # 00 00 00
    # 00 00 00
    # 00 00 00
    signal = signal.reshape(batch_size, n_frames+segments, segments, frame_step)
    # Transpose dimensions so that signal.shape = (3, 6, 2)
    # ab fg kl 00 00 00
    # cd hi mn 00 00 00
    # e0 j0 o0 00 00 00
    signal = signal.transpose(-2, -3)
    # Reshape so that signal.shape = (18, 2)
    # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0 00 00 00
    signal = torch.flatten(signal, -3, -2)
    signal.shape
    # Truncate so that signal.shape = (15, 2)
    # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0
    signal = signal[..., :(n_frames + segments - 1) * segments, :]
    # Reshape so that signal.shape = (3, 5, 2)
    # ab fg kl 00 00
    # 00 cd hi mn 00
    # 00 00 e0 j0 o0
    signal = signal.reshape(batch_size, segments, (n_frames + segments - 1), frame_step)
    # Now, reduce over the columns, to achieve the desired sum.
    signal = torch.sum(signal, -3)
    # Flatten the array.
    signal = signal.reshape(batch_size, (n_frames + segments - 1) * frame_step)

    # Truncate to final length.
    signal = signal[..., :output_length]
    return signal

def crop_and_compensate_delay(audio, audio_size, ir_size, padding, delay_compensation):
    """Copied over from ddsp
    Crop audio output from convolution to compensate for group delay.

    Args:
        audio: Audio after convolution. Tensor of shape [batch, time_steps].
        audio_size: Initial size of the audio before convolution.
        ir_size: Size of the convolving impulse response.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to compensate
        for group delay of the impulse response. If delay_compensation < 0 it
        defaults to automatically calculating a constant group delay of the
        windowed linear phase filter from frequency_impulse_response().

    Returns:
        Tensor of cropped and shifted audio.

    Raises:
        ValueError: If padding is not either 'valid' or 'same'.
    """
    # Crop the output.
    if padding == 'valid':
        crop_size = ir_size + audio_size - 1
    elif padding == 'same':
        crop_size = audio_size
    else:
        raise ValueError('Padding must be \'valid\' or \'same\', instead '
                        'of {}.'.format(padding))

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = audio.shape[-1]
    crop = total_size - crop_size
    start = ((ir_size - 1) // 2 -
            1 if delay_compensation < 0 else delay_compensation)
    end = crop - start
    return audio[:, start:-end]

def fir_filter(audio, freq_response, filter_size):
    # get IR
    h = torch.fft.irfft(freq_response, n=filter_size, dim=-1)

    # Compute filter windowed impulse response
    # window_size == filter_size
    filter_window = torch.hann_window(filter_size).roll(filter_size//2,-1).to(audio.device)
    h = filter_window[None, None, :] * h
    filtered = fft_convolve(audio, h, padding='same')
    return filtered

def fft_convolve(audio, impulse_response, padding = 'same', delay_compensation = -1):
    """ ported from ddsp original description below
    Filter audio with frames of time-varying impulse responses.

    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.

    Args:
        audio: Input audio. Tensor of shape [batch, n_samples].
        impulse_response: Finite impulse response to convolve. Can either be a 2-D
        Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
        ir_frames, ir_size]. A 2-D tensor will apply a single linear
        time-invariant filter to the audio. A 3-D Tensor will apply a linear
        time-varying filter. Automatically chops the audio into equally shaped
        blocks to match ir_frames.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (n_samples). For 'valid' the audio is
        extended to include the tail of the impulse response (n_samples +
        ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to compensate
        for group delay of the impulse response. If delay_compensation is less
        than 0 it defaults to automatically calculating a constant group delay of
        the windowed linear phase filter from frequency_impulse_response().

    Returns:
        audio_out: Convolved audio. Tensor of shape
            [batch, n_samples + ir_timesteps - 1] ('valid' padding) or shape
            [batch, audio_timesteps] ('same' padding).

    Raises:
        ValueError: If audio and impulse response have different batch size.
        ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
        number of impulse response frames is on the order of the audio size and
        not a multiple of the audio size.)
    """
    # Add a frame dimension to impulse response if it doesn't have one.
    ir_shape = impulse_response.shape
    if len(ir_shape) == 2:
        impulse_response = impulse_response[:, None, :]
        ir_shape = impulse_response.shape

    # Get shapes of audio and impulse response.
    batch_size_ir, n_ir_frames, ir_size = ir_shape
    batch_size, audio_size = audio.shape

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                        'be the same.'.format(batch_size, batch_size_ir))

    # Cut audio into frames.
    frame_size = int(np.ceil(audio_size / n_ir_frames))

    audio_frames = frame_signal(audio, frame_size)

    # Check that number of frames match.
    n_audio_frames = audio_frames.shape[1]
    if n_audio_frames != n_ir_frames:
        raise ValueError(
            'Number of Audio frames ({}) and impulse response frames ({}) do not '
            'match. For small hop size = ceil(audio_size / n_ir_frames), '
            'number of impulse response frames must be a multiple of the audio '
            'size.'.format(n_audio_frames, n_ir_frames))

    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size)

    S = torch.fft.rfft(audio_frames, n=fft_size, dim=-1) # zeropadded
    H = torch.fft.rfft(impulse_response, n=fft_size, dim=-1)

    # Multiply the FFTs (same as convolution in time).
    # Filter the original audio
    audio_ir_fft = S*H

    # Take the IFFT to resynthesize audio.
    # batch_size, n_frames, fft_size
    audio_frames_out = torch.fft.irfft(audio_ir_fft, n=fft_size, dim=-1)
    audio_out = overlap_and_add(audio_frames_out, frame_size)

    # Crop and shift the output audio.
    return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding,
                                    delay_compensation)

def pad_or_trim_to_expected_length(vector, expected_len, pad_value=0, len_tolerance=20):
    """Ported from DDSP
    Make vector equal to the expected length.

    Feature extraction functions like `compute_loudness()` or `compute_f0` produce feature vectors that vary in length depending on factors such as `sample_rate` or `hop_size`. This function corrects vectors to the expected length, warning the user if the difference between the vector and expected length was unusually high to begin with.

    Args:
        vector: Tensor. Shape [(batch,) vector_length]
        expected_len: Expected length of vector.
        pad_value: If float, value to pad at end of vector else pad_mode ('reflect', 'replicate')
        len_tolerance: Tolerance of difference between original and desired vector length.

    Returns:
        vector: Vector with corrected length.

    Raises:
        ValueError: if `len(vector)` is different from `expected_len` beyond
        `len_tolerance` to begin with.
    """
    expected_len = int(expected_len)
    vector_len = int(vector.shape[-1])

    if abs(vector_len - expected_len) > len_tolerance:
        # Ensure vector was close to expected length to begin with
        raise ValueError('Vector length: {} differs from expected length: {} '
                        'beyond tolerance of : {}'.format(vector_len,
                                                        expected_len,
                                                        len_tolerance))

    is_1d = (len(vector.shape) == 1)
    vector = vector[None, :] if is_1d else vector

    # Pad missing samples
    if vector_len < expected_len:
        n_padding = expected_len - vector_len
        if isinstance(pad_value, str):
            vector = F.pad(vector, ((0, 0, 0, n_padding)), mode=pad_value)
        else:
            vector = F.pad(vector, ((0, 0, 0, n_padding)), mode='constant', value=pad_value)
    # Trim samples
    elif vector_len > expected_len:
        vector = vector[..., :expected_len]

    # Remove temporary batch dimension.
    vector = vector[0] if is_1d else vector
    return vector

def midi_to_hz(notes):
    return 440.0 * (2.0**((notes - 69.0) / 12.0))

def hz_to_midi(frequencies):
    """torch-compatible hz_to_midi function."""
    if isinstance(frequencies, torch.Tensor):
        notes = 12.0 * (torch.log2(frequencies+1e-5) - math.log2(440.0)) + 69.0
        # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
        notes = torch.where(torch.le(frequencies, 0.0), torch.zeros_like(frequencies, device=frequencies.device), notes)
    else:
        notes = 12.0 * (math.log2(frequencies+1e-5) - math.log2(440.0)) + 69.0
    return notes

def unit_to_midi(unit, midi_min, midi_max = 90.0, clip = False):
    """Map the unit interval [0, 1] to MIDI notes."""
    unit = torch.clamp(unit, 0.0, 1.0) if clip else unit
    return midi_min + (midi_max - midi_min) * unit

def unit_to_hz(unit, hz_min, hz_max, clip = False):
    """Map unit interval [0, 1] to [hz_min, hz_max], scaling logarithmically."""
    midi = unit_to_midi(unit, midi_min=hz_to_midi(hz_min), midi_max=hz_to_midi(hz_max), clip=clip)
    return midi_to_hz(midi)

def frequencies_sigmoid(freqs, hz_min=8.2, hz_max=8000.0):
    """Sum of sigmoids to logarithmically scale network outputs to frequencies.
    without depth

    Args:
        freqs: Neural network outputs, [batch, time, n_sinusoids]
        hz_min: Lowest frequency to consider.
        hz_max: Highest frequency to consider.

    Returns:
        A tensor of frequencies in hertz [batch, time, n_sinusoids].
    """
    freqs = torch.sigmoid(freqs)
    return unit_to_hz(freqs, hz_min=hz_min, hz_max=hz_max)

# def upsample_with_windows(inputs, n_timesteps, add_endpoint):
#     """[summary]
#     code borrowed from ddsp

#     Args:
#         inputs ([type]): [description]
#         n_timesteps ([type]): [description]
#         add_endpoint ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     if len(inputs.shape) != 3:
#         raise ValueError('Upsample_with_windows() only supports 3 dimensions, '
#                         'not {}.'.format(inputs.shape))

#     # Mimic behavior of tf.image.resize.
#     # For forward (not endpointed), hold value for last interval.
#     if add_endpoint:
#         inputs = torch.cat([inputs, inputs[:, -1:, :]], axis=1)

#     n_frames = int(inputs.shape[1])
#     n_intervals = (n_frames - 1)

#     if n_frames >= n_timesteps:
#         raise ValueError('Upsample with windows cannot be used for downsampling'
#                         'More input frames ({}) than output timesteps ({})'.format(
#                             n_frames, n_timesteps))

#     if n_timesteps % n_intervals != 0.0:
#         minus_one = '' if add_endpoint else ' - 1'
#         raise ValueError(
#             'For upsampling, the target the number of timesteps must be divisible '
#             'by the number of input frames{}. (timesteps:{}, frames:{}, '
#             'add_endpoint={}).'.format(minus_one, n_timesteps, n_frames,
#                                     add_endpoint))

#     # Constant overlap-add, half overlapping windows.
#     hop_size = n_timesteps // n_intervals
#     window_length = 2 * hop_size
#     window = torch.hann_window(window_length)  # [window]

#     # Transpose for overlap_and_add.
#     x = torch.transpose(inputs, 1, 2) # [batch_size, n_channels, n_frames]

#     # Broadcast multiply.
#     # Add dimension for windows [batch_size, n_channels, n_frames, window].
#     x = x.unsqueeze(-1)
#     window = window[None, None, None, :]
#     x_windowed = (x * window)
#     x = overlap_add(x_windowed, hop_size)
#         nn.functional.fold(x_windowed, stride=hop_size)
#     # Transpose back.
#     x = tf.transpose(x, perm=[0, 2, 1])  # [batch_size, n_timesteps, n_channels]

#     # Trim the rise and fall of the first and last window.
#     return x[:, hop_size:-hop_size, :]