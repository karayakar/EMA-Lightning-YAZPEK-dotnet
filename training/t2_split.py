"""Hizalanmış uzun kayıtları kelime sürelerinden ≤10 sn / ≤250 harf parçalara böler (chunker kuralları: önce cümle
sonu, sonra virgül/;/:, sonra kelime arası). Kısa kayıtlar olduğu gibi kalır.

  .venv\\Scripts\\python.exe t2_split.py --data ../data/beyza_all
Girdi: <data>\\manifest.jsonl, durations.jsonl (tam kayıtlar)  →  Çıktı: manifest_split.jsonl, durations_split.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.acoustic import load_durations  # noqa: E402
from common.data import load_manifest  # noqa: E402
from t2_align import word_starts  # noqa: E402

MAX_FRAMES = 250  # 10 sn (25 Hz)
MAX_LETTERS = 250
SENTENCE = re.compile(r"[.!?]+[\"')]*\s*$")
CLAUSE = re.compile(r"[,;:]\s*$")
LETTER = re.compile(r"[^\W\d_]")


def finish(piece):
    """chunker.finish: parça cümle sonu noktalamasıyla biter."""
    if piece.rstrip("\"')")[-1:] in (".", "!", "?"):
        return piece
    return piece.rstrip(",;:- ") + "."


def split(text, counts):
    starts = word_starts(text) + [len(text)]
    words = [text[a:b] for a, b in zip(starts[:-1], starts[1:])]
    pieces, a = [], 0
    while a < len(words):
        b, frames, letters = a, 0, 0
        while b < len(words) and frames + counts[b] <= MAX_FRAMES and letters + len(words[b]) <= MAX_LETTERS:
            frames += counts[b]
            letters += len(words[b])
            b += 1
        if b == a:  # tek kelime sınırı aşıyor (olmamalı): kendi başına
            b = a + 1
        elif b < len(words):
            for pattern in (SENTENCE, CLAUSE):
                cut = next((k + 1 for k in range(b - 1, a, -1) if pattern.search(words[k])), None)
                if cut:
                    b = cut
                    break
        pieces.append((a, b))
        a = b
    return words, pieces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    data = Path(args.data)
    durations = load_durations(data / "durations.jsonl")
    manifest_out = open(data / "manifest_split.jsonl", "w", encoding="utf-8")
    durations_out = open(data / "durations_split.jsonl", "w", encoding="utf-8")
    kept = made = dropped = 0
    for r in load_manifest(data / "manifest.jsonl"):
        d = durations.get(r["id"])
        if d is None:
            dropped += 1
            continue
        if not r["long"] and d["frames"] <= MAX_FRAMES:
            manifest_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            durations_out.write(json.dumps(d) + "\n")
            kept += 1
            continue
        words, pieces = split(r["text_model"], d["counts"])
        offset = 0
        for k, (a, b) in enumerate(pieces):
            frames = sum(d["counts"][a:b])
            text = finish("".join(words[a:b]).strip())
            start = r["start"] + offset * 960
            offset += frames
            if not LETTER.search(text) or len(word_starts(text)) != b - a:
                dropped += 1
                continue
            piece_id = f"{r['id']}~{k}"
            manifest_out.write(json.dumps({**r, "id": piece_id, "text_model": text, "start": start,
                                           "end": start + frames * 960, "seconds": round(frames / 25, 4),
                                           "long": False, "parent": r["id"]}, ensure_ascii=False) + "\n")
            durations_out.write(json.dumps({"id": piece_id, "frames": frames, "counts": d["counts"][a:b],
                                            "score": d["score"]}) + "\n")
            made += 1
    manifest_out.close()
    durations_out.close()
    print(f"kısa (aynen) {kept}, uzunlardan parça {made}, atılan {dropped}")


if __name__ == "__main__":
    main()
