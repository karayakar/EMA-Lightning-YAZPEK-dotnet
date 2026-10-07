"""T0 manifest'inden eğitim parçaları: orijinal 24 kHz wav okunur, kırpılır, kazanç uygulanır."""
import json
import random
from pathlib import Path

import numpy as np
import torch

from .audio import read_wav

LATENT_HOP = 960  # 24 kHz'te bir latent karesi (25 Hz)


def load_manifest(path, split=None):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if split is None or record["split"] == split:
                records.append(record)
    return records


def load_segment(record):
    """Kırpılmış ve seviyesi ayarlanmış tam kayıt (float32, 24 kHz)."""
    audio, _, _ = read_wav(record["wav"])
    return audio[record["start"]:record["end"]] * np.float32(record["gain"])


class Clips(torch.utils.data.Dataset):
    """Sabit uzunlukta rastgele kesitler (uzunluk 960'ın katına yuvarlanır; kısa kayıt sıfırla doldurulur)."""

    def __init__(self, manifest, split, seconds, seed=None):
        self.records = load_manifest(manifest, split)
        self.length = max(LATENT_HOP, int(seconds * 24000) // LATENT_HOP * LATENT_HOP)
        self.seed = seed  # verilirse her öğe hep aynı kesit (doğrulama için)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        segment = load_segment(self.records[index])
        rng = random.Random(self.seed * 1_000_003 + index) if self.seed is not None else random
        if len(segment) >= self.length:
            offset = rng.randint(0, len(segment) - self.length)
            clip = segment[offset:offset + self.length]
        else:
            clip = np.zeros(self.length, np.float32)
            clip[: len(segment)] = segment
        return torch.from_numpy(np.ascontiguousarray(clip))


def write_samples(directory, name, audio, rate=24000):
    from .audio import write_wav

    Path(directory).mkdir(parents=True, exist_ok=True)
    write_wav(Path(directory) / f"{name}.wav", audio, rate)
