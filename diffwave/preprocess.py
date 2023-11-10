# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from pathlib import Path
import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

from diffwave.params import params


def transform(path, sr = 16000):
  if type(path) in [str, Path]:
    audio, sr = T.load(path)
  else:
    audio = path
  audio = torch.clamp(audio, -1.0, 1.0)

  if params.sample_rate != sr:
    print(f'Invalid sample rate {params.sample_rate} != {sr}.')
  mel_args = {
      'sample_rate': sr,
      'win_length': params.hop_samples * 4,
      'hop_length': params.hop_samples,
      'n_fft': params.n_fft,
      'f_min': 20.0,
      'f_max': sr / 2.0,
      'n_mels': params.n_mels,
      'power': 1.0,
      'normalized': True,
  }
  mel_spec_transform = TT.MelSpectrogram(**mel_args)

  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    return spectrogram


