"""T2b — CTC Viterbi zorlamalı hizalama → kelime başına kare sayısı (25 Hz), engine.piece kelime kuralıyla birebir.

  .venv\\Scripts\\python.exe t2_align.py
Çıktı: ..\\data\\male\\durations.jsonl  {id, frames, counts, score}  + t2_report.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import load_manifest  # noqa: E402
from common.mel import Mel  # noqa: E402
from t2_ctc import BLANK, CtcModel, Utterances, collate, length_batches  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAX_WORD_FRAMES = 250  # engine.py ile aynı


def word_starts(text):
    """engine.piece: kelime, boşluk olmayan ve öncesi boşluk olan karakterde başlar; boşluk önceki kelimeye aittir."""
    starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
    return [0] + starts[1:]


def viterbi(log_probs, labels):
    """CTC zorlamalı hizalama; her etiketin ilk karesi ve yol ortalama log-olasılığı. Mümkün değilse None."""
    t_len, labels = log_probs.shape[0], np.asarray(labels)
    ext = np.full(2 * len(labels) + 1, BLANK)
    ext[1::2] = labels
    s_len = len(ext)
    if t_len < len(labels):
        return None
    emit = log_probs[:, ext]  # [T, S]
    skip = np.zeros(s_len, bool)
    skip[2:] = (ext[2:] != BLANK) & (ext[2:] != ext[:-2])
    dp = np.full(s_len, -np.inf)
    dp[0], dp[1] = emit[0, 0], emit[0, 1]
    back = np.zeros((t_len, s_len), np.int8)
    for t in range(1, t_len):
        stay = dp
        step = np.concatenate(([-np.inf], dp[:-1]))
        jump = np.where(skip, np.concatenate(([-np.inf, -np.inf], dp[:-2])), -np.inf)
        best = np.stack([stay, step, jump])
        choice = best.argmax(0)
        dp = best[choice, np.arange(s_len)] + emit[t]
        back[t] = choice
    end = s_len - 1 if dp[s_len - 1] >= dp[s_len - 2] else s_len - 2
    if not np.isfinite(dp[end]):
        return None
    score = dp[end] / t_len
    state, path = end, np.empty(t_len, np.int64)
    for t in range(t_len - 1, -1, -1):
        path[t] = state
        state = int(state) - int(back[t, state])  # back int8: Python int'e çevirmezsek taşar
    first = np.full(len(labels), -1)
    for t, s in enumerate(path):
        if s % 2 == 1 and first[s // 2] < 0:
            first[s // 2] = t
    return first, float(score)


def counts_from(text, first50, frames25):
    """Harf başlangıç kareleri (50 Hz) → kelime kare sayıları (25 Hz); her kelime ≥1 kare, toplam = frames25."""
    starts = word_starts(text)
    boundaries = [0] + [int(round(first50[i] / 2)) for i in starts[1:]] + [frames25]
    for k in range(1, len(boundaries) - 1):  # tekdüze artış, her kelimeye en az 1 kare
        boundaries[k] = max(boundaries[k], boundaries[k - 1] + 1)
    for k in range(len(boundaries) - 2, 0, -1):
        boundaries[k] = min(boundaries[k], boundaries[k + 1] - 1)
    counts = np.diff(boundaries)
    if (counts < 1).any():
        return None
    return counts.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data" / "male" / "manifest.jsonl"))
    parser.add_argument("--ctc", default=str(ROOT / "checkpoints" / "male" / "ctc" / "ctc.pt"))
    parser.add_argument("--out", default=str(ROOT / "data" / "male"))
    parser.add_argument("--batch-seconds", type=float, default=600.0)
    args = parser.parse_args()

    device = torch.device("cuda")
    state = torch.load(args.ctc, map_location=device)
    vocab = state["vocab"]
    stoi = {c: i for i, c in enumerate(vocab)}
    model = CtcModel(len(vocab)).to(device).eval()
    model.load_state_dict(state["model"])
    mel = Mel().to(device)
    records = load_manifest(args.manifest)
    dataset = Utterances(records, stoi)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=length_batches(records, args.batch_seconds, 0),
                                         collate_fn=collate, num_workers=6)
    results, failures, too_long = {}, 0, 0
    done = 0
    with torch.no_grad():
        for audio, lengths, labels, label_lengths, indices in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(mel(audio.to(device)))
            log_probs = logits.float().log_softmax(-1).cpu().numpy()
            offset = 0
            for b, index in enumerate(indices):
                record = records[index]
                target = labels[offset:offset + label_lengths[b]].numpy()
                offset += label_lengths[b].item()
                frames25 = lengths[b].item() // 960
                aligned = viterbi(log_probs[b, : frames25 * 2], target)
                counts = counts_from(record["text_model"], aligned[0], frames25) if aligned else None
                if counts is None:
                    failures += 1
                    continue
                if max(counts) > MAX_WORD_FRAMES:
                    too_long += 1
                results[record["id"]] = {"id": record["id"], "frames": frames25, "counts": counts,
                                         "score": round(aligned[1], 4)}
            done += len(indices)
            if done % 10000 < len(indices):
                print(f"  {done}/{len(records)}", flush=True)

    out = Path(args.out)
    with open(out / "durations.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            if record["id"] in results:
                f.write(json.dumps(results[record["id"]]) + "\n")
    scores = np.array([r["score"] for r in results.values()])
    report = {"aligned": len(results), "failed": failures, "word_over_250_frames": too_long,
              "score_percentiles": {p: round(float(np.percentile(scores, p)), 4) for p in (0.5, 1, 5, 50)}}
    (out / "t2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
