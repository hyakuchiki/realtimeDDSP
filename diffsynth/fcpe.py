import torch
from torchfcpe import spawn_bundled_infer_model

model = spawn_bundled_infer_model(device="cuda")


def infer_fcpe(audio, hop_length, sr, f0_range=(80, 880)):
    audio_length = audio.shape[-1]
    assert audio.ndim == 1
    audio = audio[None, :, None]
    f0_target_length = (audio_length // hop_length) + 1
    # Perform pitch inference
    with torch.inference_mode():
        f0 = model.infer(
            audio,
            sr=sr,
            decoder_mode="local_argmax",  # Recommended mode
            threshold=0.006,  # Threshold for V/UV decision
            f0_min=f0_range[0],  # Minimum pitch
            f0_max=f0_range[1],  # Maximum pitch
            interp_uv=False,  # Interpolate unvoiced frames
            output_interp_target_length=f0_target_length,  # Interpolate to target length
        )
    return f0[0, :, 0].cpu()
