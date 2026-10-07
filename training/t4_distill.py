"""T4 — DMD2 damıtma: teacher (T3) → 4 adımlık öğrenci, t ∈ {0, ¼, ½, ¾}, CFG 4.0 gömülü (TRAINING_PROPOSAL T4).

Metin kodlayıcı / süre / aligner teacher'dan alınır ve dondurulur; yalnız DiT (backbone) eğitilir.
Örnekleyici ema_lightning sound_stage ile birebir (adım başına yeni gürültüyle yeniden gürültüleme).

  .venv\\Scripts\\python.exe t4_distill.py            (kaldığı yerden devam eder)
Çıktı: ..\\checkpoints\\male\\student\\  (last.pt, ema.pt [load_acoustic / export.py uyumlu], samples\\, tb\\)
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.acoustic import (Examples, collate, condition, frame_batches, load_durations,  # noqa: E402
                             predicted_timeline, rotary)
from common.data import load_manifest, write_samples  # noqa: E402
from common.evaluate import Judge  # noqa: E402

from ema_lightning.model import Acoustic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BACKBONE = ("in_proj", "t_embed", "ada_shared", "blocks", "norm_out", "ada_out", "out_proj")


def to(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def masked_mse(a, b, fmask):
    return (((a.float() - b.float()) ** 2).mean(-1) * fmask).sum() / fmask.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "male"))
    parser.add_argument("--teacher", default=str(ROOT / "checkpoints" / "male" / "teacher" / "teacher.pt"))
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "male" / "student"))
    parser.add_argument("--steps", type=int, default=8000, help="öğrenci güncelleme sayısı")
    parser.add_argument("--fake-updates", type=int, default=5, help="öğrenci adımı başına sahte skor güncellemesi")
    parser.add_argument("--max-frames", type=int, default=4000)
    parser.add_argument("--lr-student", type=float, default=2e-5)
    parser.add_argument("--lr-fake", type=float, default=5e-5)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=0, help="Windows: >0 latent sözlüğünü her işçiye kopyalar")
    parser.add_argument("--min-score", type=float, default=-0.332, help="T2 hizalama skoru alt sınırı (alt %%1)")
    parser.add_argument("--ctc", default=None, help="CER için CTC (varsayılan checkpoints/male/ctc/ctc.pt)")
    parser.add_argument("--decoder24", default=None, help="erkek decoder (yoksa orijinal 48 kHz)")
    args = parser.parse_args()

    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    source = torch.load(args.teacher, map_location="cpu", weights_only=False)
    vocab, cfg = source["vocab"], source["cfg"]
    shared = bool(cfg.get("shared_ada", True))
    stoi = {c: i for i, c in enumerate(vocab)}
    times = [float(t) for t in cfg["distilled"]["times"]]

    teacher = Acoustic(cfg, vocab, shared).to(device)
    teacher.load_state_dict(source["ema"])
    teacher.eval().requires_grad_(False)
    student = copy.deepcopy(teacher).train()
    fake = copy.deepcopy(teacher).train()
    for model in (student, fake):
        for name, p in model.named_parameters():
            p.requires_grad_(name.split(".")[0] in BACKBONE)
    student_ema = copy.deepcopy(student).eval().requires_grad_(False)
    opt_s = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr_student,
                              betas=(0.0, 0.99), weight_decay=0.0)
    opt_f = torch.optim.AdamW([p for p in fake.parameters() if p.requires_grad], lr=args.lr_fake,
                              betas=(0.0, 0.99), weight_decay=0.0)
    step = 0
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device)
        student.load_state_dict(state["student"])
        fake.load_state_dict(state["fake"])
        student_ema.load_state_dict(state["ema"])
        opt_s.load_state_dict(state["opt_s"])
        opt_f.load_state_dict(state["opt_f"])
        step = state["step"]
        print(f"devam: adım {step}")

    data = Path(args.data)
    durations = load_durations(data / "durations.jsonl")
    latents = torch.load(data / "latents.pt")
    train = Examples([r for r in load_manifest(data / "manifest.jsonl", "train") if not r["long"]], durations,
                     latents, stoi, min_score=args.min_score)
    val_records = load_manifest(data / "manifest.jsonl", "val")
    val = Examples(val_records, durations, latents, stoi)
    lengths = [len(train[i]["fw"]) for i in range(len(train))]
    judge = Judge(device, args.decoder24, args.ctc)
    texts = {r["id"]: r["text_model"] for r in val_records}
    stride = 37 if len(val) >= 8 * 37 else max(1, len(val) // 8)  # küçük val (ör. yalnız gerçek): eşit aralıklı 8
    val_batch = to(collate([val[i] for i in range(0, min(8, len(val)) * stride, stride)]), device)
    writer = SummaryWriter(out / "tb")

    @torch.no_grad()
    def prepare(batch):
        h = teacher.text(batch["ids"], batch["mask"])
        dur = teacher.chardur(h, batch["mask"])
        return condition(teacher, h, batch, dur)

    def student_x1(model, x, t, cond, fmask, cos, sin):
        tt = torch.full((x.shape[0],), t, device=device)
        return x + (1 - t) * model.backbone(x, cond, tt, fmask, cos, sin)

    def velocity(model, x, tau, cond, fmask, cos, sin):
        return model.backbone(x, cond, tau, fmask, cos, sin)

    @torch.no_grad()
    def generate(model, cond, fmask, generator=None):
        """sound_stage ile aynı 4 adım."""
        b, t_len, _ = cond.shape
        cos, sin = rotary(model, t_len, device)
        noise = torch.randn(len(times), b, t_len, model.latent_dim, device=device, generator=generator)
        x = noise[0]
        for k, t in enumerate(times):
            x1 = student_x1(model, x, t, cond, fmask, cos, sin)
            if k + 1 < len(times):
                x = (1 - times[k + 1]) * noise[k + 1] + times[k + 1] * x1
        return x1

    @torch.no_grad()
    def evaluate():
        h = teacher.text(val_batch["ids"], val_batch["mask"])
        dur = teacher.chardur(h, val_batch["mask"])
        fw, fp, fmask = predicted_timeline(dur, val_batch["cw"], val_batch["mask"])
        cond = condition(teacher, h, {**val_batch, "fw": fw, "fp": fp}, dur)
        z = generate(student_ema, cond, fmask, torch.Generator(device).manual_seed(0))
        audio = judge.decode(z, fmask)
        cer = judge.cer(audio, [texts[i] for i in val_batch["id"]])
        samples = out / "samples" / f"{step:06d}"
        for i, a in enumerate(audio):
            write_samples(samples, f"val{i}", a, judge.rate)
        (samples / "texts.txt").write_text("\n".join(texts[i] for i in val_batch["id"]), encoding="utf-8")
        writer.add_scalar("val/cer", cer, step)
        print(f"  örnek adım {step}: CER {cer:.4f} → {samples}", flush=True)

    def student_sample(cond, fmask, cos, sin):
        """Rastgele adım k; öncesi gradyansız simüle edilir (geriye simülasyon), k. adımın çıktısı gradyanlı."""
        k = random.randrange(len(times))
        x = torch.randn_like(cond[..., : student.latent_dim])
        with torch.no_grad():
            for j in range(k):
                x1 = student_x1(student, x, times[j], cond, fmask, cos, sin)
                x = (1 - times[j + 1]) * torch.randn_like(x) + times[j + 1] * x1
        return student_x1(student, x, times[k], cond, fmask, cos, sin)

    def save(path, final=False):
        if final:
            # export/load_acoustic biçimi: metin/süre/aligner teacher'dan, DiT öğrenciden (EMA)
            torch.save({"model": student_ema.state_dict(), "ema": student_ema.state_dict(),
                        "cfg": {**cfg, "distilled": {**cfg["distilled"], "method": "dmd2", "steps": len(times),
                                                     "times": times, "cfg_baked": args.guidance,
                                                     "teacher": str(args.teacher)}},
                        "vocab": vocab, "step": step}, path)
        else:
            torch.save({"student": student.state_dict(), "fake": fake.state_dict(), "ema": student_ema.state_dict(),
                        "opt_s": opt_s.state_dict(), "opt_f": opt_f.state_dict(), "step": step}, path)

    if step == 0:
        evaluate()  # damıtma öncesi: teacher ağırlıklı öğrencinin 4 adımda ne ürettiği
    started, seen, epoch = time.time(), 0, 0
    while step < args.steps:
        loader = torch.utils.data.DataLoader(train, batch_sampler=frame_batches(lengths, args.max_frames, epoch),
                                             collate_fn=collate, num_workers=args.workers, pin_memory=True)
        epoch += 1
        iterator = iter(loader)
        while step < args.steps:
            try:
                batches = [to(next(iterator), device) for _ in range(args.fake_updates)]
            except StopIteration:
                break
            # 1) sahte skor: öğrenci örneklerinde flow matching (iki zaman ölçeği)
            for batch in batches:
                cond = prepare(batch)
                fmask = batch["fmask"]
                cos, sin = rotary(fake, cond.shape[1], device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    x_hat = student_sample(cond, fmask, cos, sin).float()
                tau = torch.rand(x_hat.shape[0], device=device) * 0.96 + 0.02
                noise = torch.randn_like(x_hat)
                x_tau = (1 - tau)[:, None, None] * noise + tau[:, None, None] * x_hat
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fake_loss = masked_mse(velocity(fake, x_tau, tau, cond, fmask, cos, sin), x_hat - noise, fmask)
                opt_f.zero_grad(set_to_none=True)
                fake_loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in fake.parameters() if p.requires_grad], 1.0)
                opt_f.step()

            # 2) öğrenci: dağılım eşleme gradyanı (gerçek = teacher + CFG, sahte = fake)
            batch = batches[-1]
            cond = prepare(batch)
            fmask = batch["fmask"]
            cos, sin = rotary(student, cond.shape[1], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x_hat = student_sample(cond, fmask, cos, sin).float()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tau = torch.rand(x_hat.shape[0], device=device) * 0.96 + 0.02
                noise = torch.randn_like(x_hat)
                x_tau = (1 - tau)[:, None, None] * noise + tau[:, None, None] * x_hat
                vc = velocity(teacher, x_tau, tau, cond, fmask, cos, sin).float()
                vu = velocity(teacher, x_tau, tau, torch.zeros_like(cond), fmask, cos, sin).float()
                real = x_tau + (1 - tau)[:, None, None] * (vu + args.guidance * (vc - vu))
                fake_x1 = x_tau + (1 - tau)[:, None, None] * velocity(fake, x_tau, tau, cond, fmask, cos, sin).float()
                m = fmask[..., None].float()
                weight = ((x_hat - real).abs() * m).sum((1, 2), keepdim=True) / (m.sum((1, 2), keepdim=True) * 64)
                grad = (fake_x1 - real) / weight.clamp(min=1e-6)
                target = (x_hat - grad).detach()
            dmd_loss = 0.5 * masked_mse(x_hat, target, fmask)
            opt_s.zero_grad(set_to_none=True)
            dmd_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
            opt_s.step()
            with torch.no_grad():
                for e, p in zip(student_ema.parameters(), student.parameters()):
                    e.lerp_(p, 1 - 0.995)
            step += 1
            seen += 1
            if step % 50 == 0:
                rate = seen / (time.time() - started)
                writer.add_scalar("train/dmd", dmd_loss.item(), step)
                writer.add_scalar("train/fake", fake_loss.item(), step)
                print(f"adım {step}/{args.steps}  dmd {dmd_loss.item():.4f}  fake {fake_loss.item():.4f}  "
                      f"{rate:.2f} adım/sn  kalan ~{(args.steps - step) / rate / 60:.0f} dk", flush=True)
            if step % args.every == 0 or step == args.steps:
                evaluate()
                save(out / "last.pt")
    save(out / "ema.pt", final=True)
    print("bitti:", out / "ema.pt")


if __name__ == "__main__":
    main()
