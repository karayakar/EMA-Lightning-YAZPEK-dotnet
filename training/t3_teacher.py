"""T3 — teacher (flow matching), mevcut `ema` ağırlıklarından ince ayar; aynı mimari (TRAINING_PROPOSAL T3).

  .venv\\Scripts\\python.exe t3_teacher.py            (kaldığı yerden devam eder)
Çıktı: ..\\checkpoints\\male\\teacher\\  (last.pt, teacher.pt [load_acoustic uyumlu], samples\\, tb\\)
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.acoustic import (Examples, collate, condition, frame_batches, load_durations,  # noqa: E402
                             predicted_timeline, rotary, sample_teacher, word_sums)
from common.data import load_manifest, write_samples  # noqa: E402
from common.evaluate import Judge  # noqa: E402

from ema_lightning.model import load_acoustic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def to(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "male"))
    parser.add_argument("--weights", default=str(ROOT / "ema-lightning" / "ema.pt"))
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "male" / "teacher"))
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--max-frames", type=int, default=8000, help="batch başına (en uzun × adet) kare")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1, help="CFG için koşul düşürme olasılığı")
    parser.add_argument("--every", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=0, help="Windows: >0 latent sözlüğünü her işçiye kopyalar")
    parser.add_argument("--min-score", type=float, default=-0.332, help="T2 hizalama skoru alt sınırı (alt %%1)")
    parser.add_argument("--ctc", default=None, help="CER için CTC (varsayılan checkpoints/male/ctc/ctc.pt)")
    parser.add_argument("--decoder24", default=None, help="erkek decoder (yoksa orijinal 48 kHz)")
    parser.add_argument("--init", default=None, help="başlangıç teacher ağırlıkları (teacher.pt; 'ema' kullanılır)")
    parser.add_argument("--source", default=None, help="yalnız bu kaynaktan kayıtlar (manifest 'source', ör. r)")
    args = parser.parse_args()

    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    source = torch.load(args.weights, map_location="cpu", weights_only=False)
    vocab, cfg = source["vocab"], source["cfg"]
    stoi = {c: i for i, c in enumerate(vocab)}

    model = load_acoustic(args.weights, device).train().requires_grad_(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * 0.05)
    step = 0
    if args.init and not (out / "last.pt").exists():
        initial = torch.load(args.init, map_location=device)["ema"]
        model.load_state_dict(initial)
        ema.load_state_dict(initial)
        print(f"başlangıç: {args.init}")
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        print(f"devam: adım {step}")

    data = Path(args.data)
    durations = load_durations(data / "durations.jsonl")
    latents = torch.load(data / "latents.pt")
    def chosen(records):
        return [r for r in records if args.source is None or r.get("source") == args.source]

    train = Examples([r for r in chosen(load_manifest(data / "manifest.jsonl", "train")) if not r["long"]],
                     durations, latents, stoi, min_score=args.min_score)
    val_records = chosen(load_manifest(data / "manifest.jsonl", "val"))
    val = Examples(val_records, durations, latents, stoi)
    print(f"eğitim örneği {len(train)}, val {len(val)}")
    lengths = [len(train[i]["fw"]) for i in range(len(train))]
    judge = Judge(device, args.decoder24, args.ctc)
    texts = {r["id"]: r["text_model"] for r in val_records}
    stride = 37 if len(val) >= 8 * 37 else max(1, len(val) // 8)  # küçük val (ör. yalnız gerçek): eşit aralıklı 8
    val_batch = to(collate([val[i] for i in range(0, min(8, len(val)) * stride, stride)]), device)
    writer = SummaryWriter(out / "tb")

    def losses(batch):
        h = model.text(batch["ids"], batch["mask"])
        dur = model.chardur(h, batch["mask"])
        words = batch["counts"].shape[1]
        predicted = word_sums(dur, batch["cw"], words)
        wmask = batch["wmask"]
        dur_loss = F.mse_loss(torch.log1p(predicted[wmask]), torch.log1p(batch["counts"][wmask]))
        cond = condition(model, h, batch, dur.detach())
        keep = (torch.rand(cond.shape[0], device=device) >= args.dropout).float()[:, None, None]
        cond = cond * keep
        z = batch["z"]
        b, t_len, _ = z.shape
        t = torch.sigmoid(torch.randn(b, device=device))  # logit-normal zaman örneklemesi
        noise = torch.randn_like(z)
        x_t = (1 - t)[:, None, None] * noise + t[:, None, None] * z
        cos, sin = rotary(model, t_len, device)
        v = model.backbone(x_t, cond, t, batch["fmask"], cos, sin)
        fm = ((v.float() - (z - noise)) ** 2).mean(-1)
        fm_loss = (fm * batch["fmask"]).sum() / batch["fmask"].sum()
        return fm_loss, dur_loss

    @torch.no_grad()
    def evaluate():
        h = ema.text(val_batch["ids"], val_batch["mask"])
        dur = ema.chardur(h, val_batch["mask"])
        fw, fp, fmask = predicted_timeline(dur, val_batch["cw"], val_batch["mask"])
        batch = {**val_batch, "fw": fw, "fp": fp}
        cond = condition(ema, h, batch, dur)
        z = sample_teacher(ema, cond, fmask, generator=torch.Generator(device).manual_seed(0))
        audio = judge.decode(z, fmask)
        cer = judge.cer(audio, [texts[i] for i in val_batch["id"]])
        samples = out / "samples" / f"{step:06d}"
        for i, a in enumerate(audio):
            write_samples(samples, f"val{i}", a, judge.rate)
        (samples / "texts.txt").write_text("\n".join(texts[i] for i in val_batch["id"]), encoding="utf-8")
        writer.add_scalar("val/cer", cer, step)
        print(f"  örnek adım {step}: CER {cer:.4f} → {samples}", flush=True)

    def save(path, final=False):
        payload = {"model": model.state_dict(), "ema": ema.state_dict(), "cfg": cfg, "vocab": vocab, "step": step}
        if not final:
            payload |= {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
        torch.save(payload, path)

    started, seen, epoch = time.time(), 0, 0
    while step < args.steps:
        loader = torch.utils.data.DataLoader(train, batch_sampler=frame_batches(lengths, args.max_frames, epoch),
                                             collate_fn=collate, num_workers=args.workers, pin_memory=True)
        epoch += 1
        for batch in loader:
            batch = to(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fm_loss, dur_loss = losses(batch)
                loss = fm_loss + dur_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for e, p in zip(ema.parameters(), model.parameters()):
                    e.lerp_(p, 1 - 0.999)
            step += 1
            seen += 1
            if step % 100 == 0:
                rate = seen / (time.time() - started)
                writer.add_scalar("train/fm", fm_loss.item(), step)
                writer.add_scalar("train/dur", dur_loss.item(), step)
                print(f"adım {step}/{args.steps}  fm {fm_loss.item():.4f}  dur {dur_loss.item():.4f}  "
                      f"{rate:.1f} adım/sn  kalan ~{(args.steps - step) / rate / 60:.0f} dk", flush=True)
            if step % args.every == 0 or step == args.steps:
                ema.eval()
                evaluate()
                save(out / "last.pt")
            if step >= args.steps:
                break
    save(out / "teacher.pt", final=True)
    print("bitti:", out / "teacher.pt")


if __name__ == "__main__":
    main()
