from pathlib import Path
import numpy as np
import torch

from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave

wd = Path().parent.absolute()
model_dir = wd/'weights'/'diffwave-ljspeech.pt'
model = None

@torch.inference_mode()
def diffuser_predict(spectrogram, params = None, fast_sampling=False, device = torch.device('cuda')):
    global model
    if model is None:
        checkpoint = torch.load(model_dir, map_location=device)
        model = DiffWave(AttrDict(base_params)).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print("[DiffWave] Loaded Diffusion Model")
    model.params.override(params)
    
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
        for t in range(len(training_noise_schedule) - 1):
            if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                T.append(t + twiddle)
                break
    T = np.array(T, dtype=np.float32)


    if not model.params.unconditional:
        if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
            spectrogram = spectrogram.unsqueeze(0)
        spectrogram = spectrogram.to(device)
        audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    else:
        audio = torch.randn(1, params.audio_len, device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    for n in range(len(alpha) - 1, -1, -1):
        c1 = 1 / alpha[n]**0.5
        c2 = beta[n] / (1 - alpha_cum[n])**0.5
        audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
        if n > 0:
            noise = torch.randn_like(audio)
            sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
            audio += sigma * noise
        audio = torch.clamp(audio, -1.0, 1.0)
    return audio, model.params.sample_rate

    

