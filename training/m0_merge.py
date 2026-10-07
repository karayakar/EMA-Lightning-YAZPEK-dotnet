"""Birden çok T0 manifest'ini öneklerle birleştirir; kaynak başına yeni tohumlu val/test.

  .venv\\Scripts\\python.exe m0_merge.py --part r,../data/beyza,10 --part s,../data/beyza_synth,200 --out ../data/beyza_all
--part önek,klasör,val_test_adedi  (val ve test için ayrı ayrı bu kadar kayıt)
Çıktı: <out>\\manifest.jsonl (alanlar + "source" = önek)
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import load_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    merged = []
    for part in args.part:
        prefix, folder, count = part.split(",")
        records = load_manifest(Path(folder) / "manifest.jsonl")
        ids = sorted(r["id"] for r in records)
        random.Random(args.seed).shuffle(ids)
        k = int(count)
        split = {i: "val" for i in ids[:k]} | {i: "test" for i in ids[k:2 * k]}
        for r in records:
            merged.append({**r, "id": f"{prefix}_{r['id']}", "source": prefix, "split": split.get(r["id"], "train")})
        print(f"{prefix}: {len(records)} kayıt, val {k}, test {k}, {sum(r['seconds'] for r in records) / 3600:.2f} saat")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"→ {out / 'manifest.jsonl'}: {len(merged)} kayıt")


if __name__ == "__main__":
    main()
