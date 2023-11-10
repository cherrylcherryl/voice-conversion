from pathlib import Path
import torch
import torch.nn as nn
from diffwave.diffuser import diffuser_predict
from knn_vc.matcher import KNeighborsVC
from typing import Any, List
from diffwave.preprocess import transform

class KDiffuserVC(nn.Module):
    def __init__(
            self,
            knn_vc : KNeighborsVC | Any,
            sample_per_wave : int = 2,
            max_samples : int = 10,
    ) -> None:
        super(KDiffuserVC, self).__init__()
        self.knn_vc = knn_vc
        self.sample_per_wave = sample_per_wave
        self.max_samples = max_samples
    
    def forward(
            self
    ) -> None:
        pass

    @torch.inference_mode()
    def expand_wave(
            self,
            waves : List[str | Path | torch.Tensor],
    ) -> List[torch.Tensor]:
        preprocessed_waves = [transform(wave) for wave in waves]
        expand_set = []
        for spec in preprocessed_waves:
            for _ in range(self.sample_per_wave):
                expand_set.append(diffuser_predict(spec))
                if len(expand_set) >= self.max_samples:
                    return expand_set
        return expand_set
    
    @torch.inference_mode()
    def convert(
        self,
        src_wave : str | Path | torch.Tensor,
        ref_wave : List[str | Path | torch.Tensor]
    ) -> torch.Tensor:
        expand_set = ref_wave + self.expand_wave(ref_wave)

        query_seq = self.knn_vc.get_features(src_wave)
        matching_set = self.knn_vc.get_matching_set(expand_set)

        out_wav = self.knn_vc.match(query_seq, matching_set, topk=4)

        return out_wav

