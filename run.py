
from wavlm.WavLM import WavLM, WavLMConfig
from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from knn_vc.matcher import KNeighborsVC
from pathlib import Path

import torch
import json
import torchaudio

from vc.model import KNNVoiceExpanderVC

wd = Path().parent.absolute()
hifigan_weight = wd/'weights'/'HifiGAN-prematch.pt'
diffusion_weight = wd/'weights'/'diffwave-ljspeech.pt'
wavlm_weight = wd/'weights'/'WavLM-Large.pt'


def load_hifigan(device='cpu') -> HiFiGAN:
    with open(wd/'hifigan'/'config_v1_wavlm.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    device = torch.device(device)

    generator = HiFiGAN(h).to(device)
    
    state_dict = torch.load(hifigan_weight, map_location=device)
    
    generator.load_state_dict(state_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    print(f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters.")
    return generator, h

def load_wavlm(device='cpu') -> WavLM:
    """Load the WavLM large checkpoint from the original paper. See https://github.com/microsoft/unilm/tree/master/wavlm for details. """
    checkpoint = torch.load(wavlm_weight)
    
    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"[WavLM] WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
    return model

def load_knn_vc(device='cpu')-> KNeighborsVC:
    """ Load kNN-VC (WavLM encoder and HiFiGAN decoder). Optionally use vocoder trained on `prematched` data. """
    hifigan, hifigan_cfg = load_hifigan(device)
    wavlm = load_wavlm(device)
    knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device)
    return knnvc


if __name__ == "__main__":

    # import argparse

    # def path_list(arg):
    #     return arg.split(',')

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", "-i", type=str, help="Input file 16KHz to convert")
    # parser.add_argument("--reference", "-r", type=path_list, help="Reference voice convert to")
    # parser.add_argument("--output", "-o", type=str, help="Output file name")
    # args = parser.parse_args()

    corpus = []
    with open("sample_data.txt", "r") as f:
        for line in f:
            corpus.append(line.strip("\n"))

    knn_vc = load_knn_vc('cuda' if torch.cuda.is_available() else 'cpu')
    vc = KNNVoiceExpanderVC(knn_vc, 2, corpus)
    output_wav = vc.convert("audio.wav", ["demo.wav"])

    torchaudio.save("vc.wav", output_wav[None], 16000)

