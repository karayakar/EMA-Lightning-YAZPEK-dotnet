"""Beyza sentetik veri: tohumlu rastgele ~N saatlik alt küme, 48 → 24 kHz, wavs\\ + metadata.csv (kaynağa dokunulmaz).

  .venv\\Scripts\\python.exe s0_synth_subset.py
Çıktı: ..\\data\\beyza_synth_src\\wavs\\*.wav (24 kHz PCM16), metadata.csv ("ad.wav| metin")
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.audio import read_wav, write_wav  # noqa: E402
from common.mel import Downsample2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def convert(job):
    """Tek dosya: 48 → 24 kHz (varsa atlanır). (ad, saniye) ya da (ad, None)."""
    source, out, name = job
    target = out / "wavs" / name
    if target.exists():
        with open(target, "rb") as f:
            return name, (target.stat().st_size - 44) / 2 / 24000
    try:
        audio, rate, _ = read_wav(source / "audiosINTPBeyza" / name)
    except Exception:
        return name, None
    if rate == 48000:
        with torch.no_grad():
            audio = DOWN(torch.from_numpy(audio)[None])[0].numpy()
    elif rate != 24000:
        return name, None
    write_wav(target, np.asarray(audio, np.float32), 24000)
    return name, len(audio) / 24000


DOWN = Downsample2()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=r"G:\OPENAI\Turkish_VoiceDatasets\audiosINTPBeyza")
    parser.add_argument("--metadata", default="metadata130k.csv")
    parser.add_argument("--out", default=str(ROOT / "data" / "beyza_synth_src"))
    parser.add_argument("--count", type=int, default=26500, help="ortalama 13.6 sn ile ~100 saat")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from concurrent.futures import ProcessPoolExecutor

    torch.set_num_threads(1)
    source = Path(args.source)
    entries = []
    with open(source / args.metadata, encoding="utf-8", errors="replace") as f:
        for line in f:
            name, _, text = line.rstrip("\r\n").partition("|")
            if name.strip() and text.strip():
                entries.append((name.strip(), text.strip()))
    random.Random(args.seed).shuffle(entries)
    entries = entries[: args.count]
    texts = dict(entries)
    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    seconds, written, skipped = 0.0, 0, 0
    with ProcessPoolExecutor(args.workers) as pool, \
            open(out / "metadata.csv", "w", encoding="utf-8", newline="\n") as meta:
        for name, duration in pool.map(convert, [(source, out, n) for n, _ in entries], chunksize=32):
            if duration is None:
                skipped += 1
                continue
            meta.write(f"{name}| {texts[name]}\n")
            seconds += duration
            written += 1
            if written % 2000 == 0:
                print(f"  {written} dosya, {seconds / 3600:.1f} saat", flush=True)
    print(f"bitti: {written} dosya, {seconds / 3600:.2f} saat, atlanan {skipped} → {out}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
