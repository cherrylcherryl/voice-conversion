from pathlib import Path
import torch
import torch.nn as nn
from knn_vc.matcher import KNeighborsVC
from typing import Any, List
from voice_cloning.inference import synthesis_voice
import random
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class KNNVoiceExpanderVC(nn.Module):
    def __init__(
            self,
            knn_vc : KNeighborsVC | Any,
            max_sample: int = 2,
            corpus : List[str] = []
    ) -> None:
        super(KNNVoiceExpanderVC, self).__init__()
        self.knn_vc = knn_vc
        self.max_sample = max_sample
        self.corpus = corpus

    def forward(self) -> None: 
        pass

    @torch.inference_mode()
    def expand_wave(
            self,
            waves : List[str],
    ) -> List[torch.Tensor]:
        expand_set = []
        for wave in waves:
            rnd = random.randint(0, len(self.corpus) - 1)
            text_sample = self.corpus[rnd]
            wav_syn, sr = synthesis_voice(wave, 16000, text_sample)
            expand_set.append(torch.from_numpy(wav_syn.astype(np.float32)).to(device))
        return expand_set
    
    @torch.inference_mode()
    def convert(
        self,
        src_wave : str,
        ref_wave : List[str]
    ) -> torch.Tensor:
        expand_set = ref_wave + self.expand_wave(ref_wave)

        query_seq = self.knn_vc.get_features(src_wave)
        matching_set = self.knn_vc.get_matching_set(expand_set)

        out_wav = self.knn_vc.match(query_seq, matching_set, topk=4)

        return out_wav
