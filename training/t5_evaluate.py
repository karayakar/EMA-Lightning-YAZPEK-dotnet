"""T5 — test seti değerlendirmesi: noktalamasız CER (kendi CTC'miz), teacher (16 adım) / öğrenci (4 adım) / gerçek kayıt.

  .venv\\Scripts\\python.exe t5_evaluate.py
Çıktı: ..\\checkpoints\\male\\t5_report.json + samples\\test\\
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.acoustic import Examples, collate, condition, load_durations, predicted_timeline, rotary  # noqa: E402
from common.data import load_manifest, load_segment, write_samples  # noqa: E402
from common.evaluate import Judge  # noqa: E402
from t2_ctc import edit_distance, greedy  # noqa: E402

from ema_lightning.model import load_acoustic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LETTERS = re.compile(r"[^\w ]|_|\d")


def plain(text):
    """Noktalama ve fazla boşluk atılır; yalnız harfler ve tek boşluk."""
    return re.sub(r"\s+", " ", LETTERS.sub(" ", text)).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "male"))
    parser.add_argument("--teacher", default=str(ROOT / "checkpoints" / "male" / "teacher" / "teacher.pt"))
    parser.add_argument("--student", default=str(ROOT / "checkpoints" / "male" / "student" / "ema.pt"))
    parser.add_argument("--decoder24", default=str(ROOT / "checkpoints" / "male" / "decoder" / "decoder24.pt"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "male"))
    parser.add_argument("--ctc", default=None, help="CER için CTC (varsayılan checkpoints/male/ctc/ctc.pt)")
    parser.add_argument("--source", default=None, help="yalnız bu kaynaktan test kayıtları (manifest 'source', ör. r)")
    args = parser.parse_args()

    device = torch.device("cuda")
    judge = Judge(device, args.decoder24, args.ctc)
    stoi = judge.stoi
    data = Path(args.data)
    records = [r for r in load_manifest(data / "manifest.jsonl", "test")
               if args.source is None or r.get("source") == args.source]
    by_id = {r["id"]: r for r in records}
    tests = Examples(records, load_durations(data / "durations.jsonl"), torch.load(data / "latents.pt"), stoi)
    items = [tests[i] for i in range(min(args.count, len(tests)))]

    def cer_plain(audio, texts):
        errors = chars = 0
        for a, text in zip(audio, texts):
            x = torch.from_numpy(a[: len(a) // 960 * 960]).to(device)[None]
            with torch.no_grad():
                log_probs = judge.ctc(judge.mel(x)).float().log_softmax(-1)[0].cpu()
            hyp = plain("".join(judge.vocab[k] for k in greedy(log_probs, log_probs.shape[0])))
            ref = plain(text)
            errors += edit_distance(list(hyp), list(ref))
            chars += len(ref)
        return errors / max(chars, 1)

    report = {"test_sentences": len(items)}
    texts = [by_id[x["id"]]["text_model"] for x in items]
    real = [load_segment(by_id[x["id"]]) for x in items]
    report["cer_recordings"] = round(cer_plain(real, texts), 4)

    def synthesize(model, sampler, name):
        audio, seconds, started = [], 0.0, time.time()
        for i in range(0, len(items), 16):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in collate(items[i:i + 16]).items()}
            with torch.no_grad():
                h = model.text(batch["ids"], batch["mask"])
                dur = model.chardur(h, batch["mask"])
                fw, fp, fmask = predicted_timeline(dur, batch["cw"], batch["mask"])
                cond = condition(model, h, {**batch, "fw": fw, "fp": fp}, dur)
                z = sampler(model, cond, fmask)
            audio += judge.decode(z, fmask)
        elapsed = time.time() - started
        seconds = sum(len(a) for a in audio) / judge.rate
        for i, a in enumerate(audio[:12]):
            write_samples(Path(args.out) / "samples" / "test" / name, f"test{i}", a, judge.rate)
        report[f"cer_{name}"] = round(cer_plain(audio, texts), 4)
        report[f"gpu_rtf_{name}"] = round(elapsed / seconds, 4)
        print(name, report[f"cer_{name}"], flush=True)

    from common.acoustic import sample_teacher

    teacher = load_acoustic(args.teacher, device)
    synthesize(teacher, lambda m, c, f: sample_teacher(m, c, f, generator=torch.Generator(device).manual_seed(0)),
               "teacher16")
    student = load_acoustic(args.student, device)

    def four_steps(model, cond, fmask):
        b, t_len, _ = cond.shape
        cos, sin = rotary(model, t_len, device)
        generator = torch.Generator(device).manual_seed(0)
        noise = torch.randn(len(model.times), b, t_len, model.latent_dim, device=device, generator=generator)
        x = noise[0]
        for k, t in enumerate(model.times):
            v = model.backbone(x, cond, torch.full((b,), t, device=device), fmask, cos, sin)
            x1 = x + (1 - t) * v
            if k + 1 < len(model.times):
                x = (1 - model.times[k + 1]) * noise[k + 1] + model.times[k + 1] * x1
        return x1

    synthesize(student, four_steps, "student4")
    (Path(args.out) / "t5_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
