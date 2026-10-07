"""Değerlendirme: latent → ses (erkek 24 kHz decoder; yoksa orijinal 48 kHz decoder + ×2 alçaltma) ve CTC ile CER."""
from pathlib import Path

import torch

from ema_lightning.decoder import Decoder

from .mel import Downsample2, Mel
from .nets import DECODER24, frozen_decoder48

ROOT = Path(__file__).resolve().parent.parent.parent


class Judge:
    def __init__(self, device, decoder24=None, ctc=None):
        from t2_ctc import CtcModel  # training klasörü sys.path'te

        self.device = device
        self.rate = 24000
        decoder24 = Path(decoder24 or ROOT / "checkpoints" / "male" / "decoder" / "decoder24.pt")
        if decoder24.exists():
            self.decoder = Decoder(**DECODER24).to(device).eval()
            self.decoder.load_state_dict(torch.load(decoder24, map_location=device)["G"])
            self.down = None
        else:
            self.decoder = frozen_decoder48(ROOT / "ema-lightning" / "decoder.pt", device)
            self.down = Downsample2().to(device)
        self.mel = Mel().to(device)
        state = torch.load(ctc or ROOT / "checkpoints" / "male" / "ctc" / "ctc.pt", map_location=device)
        self.vocab = state["vocab"]
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.ctc = CtcModel(len(self.vocab)).to(device).eval()
        self.ctc.load_state_dict(state["model"])

    @torch.no_grad()
    def decode(self, z, fmask):
        """z [B, T, 64], fmask [B, T] → her örnek kendi uzunluğunda numpy ses (24 kHz)."""
        audio = []
        for i in range(z.shape[0]):
            frames = int(fmask[i].sum())
            y = self.decoder(z[i: i + 1, :frames].float().transpose(1, 2))
            if self.down is not None:
                y = self.down(y)
            audio.append(y[0, : frames * 960].clamp(-1, 1).cpu().numpy())
        return audio

    @torch.no_grad()
    def cer(self, audio, texts):
        from t2_ctc import edit_distance, greedy

        errors = chars = 0
        for a, text in zip(audio, texts):
            x = torch.from_numpy(a[: len(a) // 960 * 960]).to(self.device)[None]
            log_probs = self.ctc(self.mel(x)).float().log_softmax(-1)[0].cpu()
            target = [self.stoi.get(c, 1) for c in text]
            errors += edit_distance(greedy(log_probs, log_probs.shape[0]), target)
            chars += len(target)
        return errors / max(chars, 1)
