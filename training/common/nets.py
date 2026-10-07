"""T1 ağları: latent encoder, 24 kHz decoder kurucusu, MPD + MRD discriminator'lar (tasarım: TRAINING_PROPOSAL T1)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from ema_lightning.decoder import Decoder, load_decoder

DECODER24 = dict(latent_dim=64, ch=256, rates=(8, 6, 5, 2, 2), kernels=(16, 12, 10, 4, 4), rb_kernels=(3, 5, 9),
                 rb_dilations=((1, 3, 5),) * 3)


class ConvNeXt1d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, 7, padding=3, groups=ch)
        self.norm = nn.LayerNorm(ch)
        self.pw1 = nn.Linear(ch, 4 * ch)
        self.pw2 = nn.Linear(4 * ch, ch)
        self.gamma = nn.Parameter(torch.full((ch,), 1e-2))

    def forward(self, x):
        y = self.norm(self.dw(x).transpose(1, 2))
        return x + (self.gamma * self.pw2(F.gelu(self.pw1(y)))).transpose(1, 2)


class Encoder(nn.Module):
    """log-mel [B, 100, T·4] (100 kare/sn) → latent [B, 64, T] (25 Hz)."""

    def __init__(self, n_mels=100, ch=256, latent_dim=64, blocks=3):
        super().__init__()
        self.pre = nn.Conv1d(n_mels, ch, 7, padding=3)
        self.stage1 = nn.Sequential(*(ConvNeXt1d(ch) for _ in range(blocks)))
        self.down1 = nn.Conv1d(ch, ch, 4, stride=2, padding=1)
        self.stage2 = nn.Sequential(*(ConvNeXt1d(ch) for _ in range(blocks)))
        self.down2 = nn.Conv1d(ch, ch, 4, stride=2, padding=1)
        self.stage3 = nn.Sequential(*(ConvNeXt1d(ch) for _ in range(blocks)))
        self.post = nn.Conv1d(ch, latent_dim, 3, padding=1)

    def forward(self, mel):
        x = self.stage1(self.pre(mel))
        x = self.stage2(self.down1(x))
        x = self.stage3(self.down2(x))
        return self.post(x)


def frozen_decoder48(path, device):
    decoder = load_decoder(str(path), device)
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder.eval()


def decoder24_from(path):
    """24 kHz decoder (hop 960): orijinalin ilk 5 yükseltme katmanı + resblock'ları kopyalanır, `post` yeniden başlar."""
    source = load_decoder(str(path), "cpu").state_dict()
    decoder = Decoder(**DECODER24)
    own = decoder.state_dict()
    copied, skipped = [], []
    for key, value in own.items():
        if key in source and source[key].shape == value.shape:
            own[key] = source[key].clone()
            copied.append(key)
        else:
            skipped.append(key)
    decoder.load_state_dict(own)
    return decoder, copied, skipped


class PeriodDiscriminator(nn.Module):
    # HiFi-GAN kanalları; bellek darsa (1, 32, 64, 128, 256, 256) ile daraltılabilir
    def __init__(self, period, channels=(1, 32, 128, 512, 1024, 1024)):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(channels[i], channels[i + 1], (5, 1), (3 if i < 4 else 1, 1), padding=(2, 0)))
            for i in range(5))
        self.post = weight_norm(nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, x):
        b, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        x = x.view(b, 1, -1, self.period)
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.post(x)
        features.append(x)
        return x.flatten(1), features


class ResolutionDiscriminator(nn.Module):
    def __init__(self, n_fft, hop, win):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        spec = torch.stft(x.float(), self.n_fft, self.hop, self.win, window=self.window, return_complex=True).abs()
        x = spec[:, None]
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.post(x)
        features.append(x)
        return x.flatten(1), features


class Discriminators(nn.Module):
    """HiFi-GAN MPD (periyot 2, 3, 5, 7, 11) + UnivNet MRD (üç çözünürlük)."""

    def __init__(self):
        super().__init__()
        self.items = nn.ModuleList([PeriodDiscriminator(p) for p in (2, 3, 5, 7, 11)] +
                                   [ResolutionDiscriminator(*r) for r in ((1024, 120, 600), (2048, 240, 1200),
                                                                          (512, 50, 240))])

    def forward(self, x):
        return [d(x) for d in self.items]


def discriminator_loss(real, fake):
    loss = 0.0
    for (r, _), (f, _) in zip(real, fake):
        loss = loss + torch.mean((1 - r) ** 2) + torch.mean(f ** 2)
    return loss


def generator_losses(real, fake):
    adversarial = sum(torch.mean((1 - f) ** 2) for f, _ in fake)
    matching = 0.0
    for (_, real_features), (_, fake_features) in zip(real, fake):
        for r, f in zip(real_features, fake_features):
            matching = matching + F.l1_loss(f, r.detach())
    return adversarial, matching
