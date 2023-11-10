import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

wd = Path().parent.absolute()
encoder_dir = wd/'weights'/'encoder.pt'
synthesis_dir = wd/'weights'/'synthesizer.pt'
vocoder_dir = wd/'weights'/'vocoder.pt'

encoder.load_model(encoder_dir)
synthesizer = Synthesizer(synthesis_dir)
vocoder.load_model(vocoder_dir)

def synthesis_voice(wave: str | np.ndarray, sampling_rate : int, content : str):
    if isinstance(wave, str):
        original_wav, sampling_rate = librosa.load(str(wave))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    if isinstance(wave, np.ndarray):
        preprocessed_wav = encoder.preprocess_wav(wave, sampling_rate)
    embed = encoder.embed_utterance(preprocessed_wav)

    texts = [content]
    embeds = [embed]

    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]

    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav)

    return generated_wav.astype(np.float32)



