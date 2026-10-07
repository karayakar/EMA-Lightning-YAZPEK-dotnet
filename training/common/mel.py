"""24 kHz log-mel, ×2 seyreltme (48 → 24 kHz) ve spektral loss'lar (GPU, torch)."""
import math

import torch
import torch.nn.functional as F

RATE = 24000
N_FFT = 1024
HOP = 240  # 100 kare/sn; 4 kare = 1 latent (25 Hz, 960 örnek)
N_MELS = 100


def _mel_filters(n_fft, n_mels, rate, fmin=0.0, fmax=None):
    """HTK mel ölçeği, alan normalize üçgen filtreler [n_mels, n_fft/2+1]."""
    fmax = fmax or rate / 2
    mel = lambda f: 2595.0 * math.log10(1.0 + f / 700.0)  # noqa: E731
    points = torch.linspace(mel(fmin), mel(fmax), n_mels + 2, dtype=torch.float64)
    hz = 700.0 * (10 ** (points / 2595.0) - 1.0)
    bins = torch.fft.rfftfreq(n_fft, 1 / rate).double()
    filters = torch.zeros(n_mels, len(bins), dtype=torch.float64)
    for m in range(n_mels):
        lo, center, hi = hz[m], hz[m + 1], hz[m + 2]
        up = (bins - lo) / (center - lo)
        down = (hi - bins) / (hi - center)
        filters[m] = torch.clamp(torch.minimum(up, down), min=0) * (2.0 / (hi - lo))
    return filters.float()


class Mel(torch.nn.Module):
    """[B, L] (L, 960'ın katı) → log-mel [B, 100, L/240]."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("filters", _mel_filters(N_FFT, N_MELS, RATE), persistent=False)

    def forward(self, audio):
        spec = torch.stft(audio.float(), N_FFT, HOP, window=self.window, center=True, return_complex=True).abs()
        spec = spec[..., : audio.shape[-1] // HOP]  # center=True'nun fazladan son karesi atılır
        return torch.log(torch.clamp(self.filters @ spec, min=1e-5))


class Downsample2(torch.nn.Module):
    """48 → 24 kHz: Kaiser pencereli sinc alçak geçiren (24 kHz Nyquist'in %95'i) + 2'de 1 seçme."""

    def __init__(self, half_taps=64, beta=8.0):
        super().__init__()
        k = torch.arange(-half_taps, half_taps + 1, dtype=torch.float64)
        cutoff = 0.95 * 0.25  # devir/örnek, 48 kHz'te
        taps = torch.special.sinc(2 * cutoff * k) * 2 * cutoff
        taps *= torch.kaiser_window(2 * half_taps + 1, periodic=False, beta=beta, dtype=torch.float64)
        self.register_buffer("taps", (taps / taps.sum()).float()[None, None], persistent=False)
        self.pad = half_taps

    def forward(self, audio):
        return F.conv1d(audio[:, None].float(), self.taps, stride=2, padding=self.pad)[:, 0]


class MultiResolutionStft(torch.nn.Module):
    """Spektral yakınsama + log genlik L1, üç çözünürlükte."""

    def __init__(self, resolutions=((512, 128), (1024, 256), (2048, 512))):
        super().__init__()
        self.resolutions = resolutions
        for n_fft, _ in resolutions:
            self.register_buffer(f"w{n_fft}", torch.hann_window(n_fft), persistent=False)

    def forward(self, prediction, target):
        loss = 0.0
        for n_fft, hop in self.resolutions:
            window = getattr(self, f"w{n_fft}")
            p = torch.stft(prediction.float(), n_fft, hop, window=window, return_complex=True).abs().clamp(min=1e-7)
            t = torch.stft(target.float(), n_fft, hop, window=window, return_complex=True).abs().clamp(min=1e-7)
            loss = loss + torch.linalg.norm(t - p) / torch.linalg.norm(t).clamp(min=1e-7)
            loss = loss + F.l1_loss(torch.log(p), torch.log(t))
        return loss / len(self.resolutions)
