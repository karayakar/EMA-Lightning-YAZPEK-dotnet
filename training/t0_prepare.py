"""T0 — veri hazırlığı (TRAINING_PROPOSAL §3 T0). Ses kopyalanmaz; manifest + rapor üretir.

Çalıştırma (training klasöründen):
  .venv\\Scripts\\python.exe t0_prepare.py
Çıktı: ..\\data\\male\\manifest.jsonl, t0_report.json, t0_report.md
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.audio import RATE, band_edge, gain_for, read_wav, trim  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIN_SECONDS = 0.5
MAX_SECONDS = 10.0  # chunker sınırı (~10 sn / 250 harf)
LETTER = re.compile(r"[^\W\d_]")

_frontend = None
_chunk = None


def _init(vocab):
    # her işçi süreç kendi frontend'ini kurar (çıkarımdakiyle aynı: normalizer-tr fallback + alfabe)
    global _frontend, _chunk
    from ema_lightning.chunker import chunk
    from ema_lightning.frontend import Frontend
    _frontend = Frontend(vocab)
    _chunk = chunk


def _process(item):
    uid, path, text = item
    record = {"id": uid, "wav": str(path), "text_raw": text}
    if text is None:
        return {**record, "reason": "no_text"}
    if not path.exists():
        return {**record, "reason": "no_wav"}
    try:
        audio, rate, channels = read_wav(path)
    except Exception as error:  # okunamayan dosya raporlanır, iş durmaz
        return {**record, "reason": "unreadable", "detail": str(error)}
    if rate != RATE or channels != 1:
        return {**record, "reason": "format", "detail": f"{rate} Hz, {channels} ch"}
    start, end = trim(audio, rate)
    if end <= start:
        return {**record, "reason": "silent"}
    segment = audio[start:end]
    seconds = (end - start) / rate
    spoken = _frontend(text)
    if not LETTER.search(spoken):
        return {**record, "reason": "empty_text", "text_model": spoken}
    pieces = _chunk(spoken, 1.0)
    single = len(pieces) == 1
    text_model = pieces[0][0] if single else spoken
    record.update(
        text_model=text_model,
        start=int(start),
        end=int(end),
        gain=round(float(gain_for(segment, rate)), 6),
        seconds=round(seconds, 4),
        band_hz=round(band_edge(segment, rate)),
        chars_per_second=round(len(text_model) / seconds, 3),
        long=(not single) or seconds > MAX_SECONDS,
    )
    if seconds < MIN_SECONDS:
        record["reason"] = "too_short"
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"G:\OPENAI\Turkish_VoiceDatasets\100HRSKARAY")
    parser.add_argument("--out", default=str(ROOT / "data" / "male"))
    parser.add_argument("--weights", default=str(ROOT / "ema-lightning" / "ema.pt"))
    parser.add_argument("--seed", type=int, default=20261007)
    # Windows ProcessPoolExecutor sınırı 61; iş ağırlıkla disk okuması
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()

    import torch

    vocab = torch.load(args.weights, map_location="cpu", weights_only=False)["vocab"]
    data = Path(args.data)
    texts = {}
    duplicates = 0
    with open(data / "metadata.csv", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            name, _, text = line.partition("|")
            name = name.strip()
            duplicates += name in texts
            texts[name] = text.strip()
    wavs = {p.name: p for p in (data / "wavs").glob("*.wav")}
    names = sorted(set(texts) | set(wavs), key=lambda n: (len(n), n))
    items = [(Path(n).stem, wavs.get(n, data / "wavs" / n), texts.get(n)) for n in names]
    print(f"metadata: {len(texts)} satır ({duplicates} tekrar), wav: {len(wavs)}, birleşik: {len(items)}")

    records = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(vocab,)) as pool:
        for i, record in enumerate(pool.map(_process, items, chunksize=64), 1):
            records.append(record)
            if i % 5000 == 0:
                print(f"  {i}/{len(items)}", flush=True)

    usable = [r for r in records if "reason" not in r]
    # yeni val/test: kısa (tek parça) kullanılabilir kayıtlardan tohumlu %1 + %1
    pool_ids = sorted(r["id"] for r in usable if not r["long"])
    random.Random(args.seed).shuffle(pool_ids)
    k = max(1, len(pool_ids) // 100)
    split = {uid: "val" for uid in pool_ids[:k]} | {uid: "test" for uid in pool_ids[k:2 * k]}
    for r in usable:
        r["split"] = split.get(r["id"], "train")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in usable:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "t0_rejected.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            if "reason" in r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    seconds = np.array([r["seconds"] for r in usable])
    cps = np.array([r["chars_per_second"] for r in usable])
    band = np.array([r["band_hz"] for r in usable])
    lo, hi = np.percentile(cps, [0.5, 99.5])
    outliers = sorted((r for r in usable if not lo <= r["chars_per_second"] <= hi),
                      key=lambda r: r["chars_per_second"])
    report = {
        "metadata_lines": len(texts),
        "metadata_duplicates": duplicates,
        "wav_files": len(wavs),
        "rejected": dict(Counter(r["reason"] for r in records if "reason" in r)),
        "usable": len(usable),
        "usable_hours": round(float(seconds.sum()) / 3600, 2),
        "long": int(sum(r["long"] for r in usable)),
        "split": dict(Counter(r["split"] for r in usable)),
        "seconds_percentiles": {p: round(float(np.percentile(seconds, p)), 2) for p in (1, 10, 50, 90, 99)},
        "band_hz": {"<8k": int((band < 8000).sum()), "8-10k": int(((band >= 8000) & (band < 10000)).sum()),
                    ">=10k": int((band >= 10000).sum()), "median": int(np.median(band))},
        "chars_per_second": {"p0.5": round(float(lo), 2), "median": round(float(np.median(cps)), 2),
                             "p99.5": round(float(hi), 2), "outliers": len(outliers)},
    }
    (out / "t0_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# T0 raporu", "", "```json", json.dumps(report, ensure_ascii=False, indent=2), "```", "",
             "## Konuşma hızı aykırı (chars/sn, p0.5–p99.5 dışı) — ilk/son 25", "",
             "| id | sn | chars/sn | metin |", "|---|---|---|---|"]
    for r in outliers[:25] + outliers[-25:]:
        lines.append(f"| {r['id']} | {r['seconds']} | {r['chars_per_second']} | {r['text_model'][:80]} |")
    lines += ["", "## Dar bant (< 8 kHz) — ilk 50", "", "| id | band_hz | sn |", "|---|---|---|"]
    for r in sorted((r for r in usable if r["band_hz"] < 8000), key=lambda r: r["band_hz"])[:50]:
        lines.append(f"| {r['id']} | {r['band_hz']} | {r['seconds']} |")
    (out / "t0_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
