"""T1b — 24 kHz erkek decoder'ı (hop 960), GAN ince ayar; T1a encoder dondurulmuş (TRAINING_PROPOSAL T1b).

  .venv\\Scripts\\python.exe t1_decoder.py            (kaldığı yerden devam eder)
Çıktı: ..\\checkpoints\\male\\decoder\\  (last.pt, decoder24.pt, samples\\, tb\\)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import Clips, write_samples  # noqa: E402
from common.mel import Mel, MultiResolutionStft  # noqa: E402
from common.nets import (DECODER24, Discriminators, Encoder, decoder24_from, discriminator_loss,  # noqa: E402
                         generator_losses)

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data" / "male" / "manifest.jsonl"))
    parser.add_argument("--weights", default=str(ROOT / "ema-lightning"))
    parser.add_argument("--encoder", default=str(ROOT / "checkpoints" / "male" / "encoder" / "encoder.pt"))
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "male" / "decoder"))
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=0.48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2000, help="ilk adımlar yalnız mel+STFT (GAN yok)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--every", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load(args.encoder, map_location=device)["encoder"])
    encoder.eval().requires_grad_(False)
    generator, copied, skipped = decoder24_from(Path(args.weights) / "decoder.pt")
    generator = generator.to(device)
    print(f"decoder24: {len(copied)} tensör kopyalandı, yeni başlatılan: {skipped}")
    discriminators = Discriminators().to(device)
    mel, mrstft = Mel().to(device), MultiResolutionStft().to(device)

    opt_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)
    opt_d = torch.optim.AdamW(discriminators.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, args.steps, eta_min=args.lr * 0.1)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, args.steps, eta_min=args.lr * 0.1)
    step = 0
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device)
        generator.load_state_dict(state["generator"])
        discriminators.load_state_dict(state["discriminators"])
        for obj, key in ((opt_g, "opt_g"), (opt_d, "opt_d"), (sched_g, "sched_g"), (sched_d, "sched_d")):
            obj.load_state_dict(state[key])
        step = state["step"]
        print(f"devam: adım {step}")

    train = torch.utils.data.DataLoader(Clips(args.manifest, "train", args.seconds), batch_size=args.batch,
                                        shuffle=True, num_workers=args.workers, drop_last=True,
                                        persistent_workers=True, pin_memory=True)
    val = next(iter(torch.utils.data.DataLoader(Clips(args.manifest, "val", 4.8, seed=1), batch_size=6)))
    writer = SummaryWriter(out / "tb")

    def generate(x):
        with torch.no_grad():
            z = encoder(mel(x))
        return generator(z)[:, : x.shape[1]]

    @torch.no_grad()
    def evaluate():
        generator.eval()
        x = val.to(device)
        y = generate(x)
        mel_loss = F.l1_loss(mel(y), mel(x)).item()
        writer.add_scalar("val/mel", mel_loss, step)
        samples = out / "samples" / f"{step:06d}"
        for i in range(x.shape[0]):
            write_samples(samples, f"val{i}_orig", x[i].cpu().numpy())
            write_samples(samples, f"val{i}_recon", y[i].cpu().numpy())
        generator.train()
        print(f"  val adım {step}: mel {mel_loss:.4f} → {samples}", flush=True)
        return mel_loss

    generator.train()
    started, seen = time.time(), 0
    while step < args.steps:
        for x in train:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = generate(x).float()
            adversarial_on = step >= args.warmup

            if adversarial_on:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss_d = discriminator_loss(discriminators(x), discriminators(y.detach()))
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminators.parameters(), 10.0)
                opt_d.step()

            mel_loss = F.l1_loss(mel(y), mel(x))
            stft_loss = mrstft(y, x)
            loss_g = 45 * mel_loss + stft_loss
            if adversarial_on:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    adversarial, matching = generator_losses(discriminators(x), discriminators(y))
                loss_g = loss_g + adversarial + 2 * matching
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
            opt_g.step()
            sched_g.step()
            sched_d.step()
            step += 1
            seen += 1
            if step % 100 == 0:
                rate = seen / (time.time() - started)
                writer.add_scalar("train/mel", mel_loss.item(), step)
                writer.add_scalar("train/stft", stft_loss.item(), step)
                if adversarial_on:
                    writer.add_scalar("train/d", loss_d.item(), step)
                    writer.add_scalar("train/adv", adversarial.item(), step)
                print(f"adım {step}/{args.steps}  mel {mel_loss.item():.4f}  stft {stft_loss.item():.4f}"
                      + (f"  d {loss_d.item():.3f}  adv {adversarial.item():.3f}" if adversarial_on else "  (ısınma)")
                      + f"  {rate:.2f} adım/sn  kalan ~{(args.steps - step) / rate / 60:.0f} dk", flush=True)
            if step % args.every == 0 or step == args.steps:
                evaluate()
                torch.save({"generator": generator.state_dict(), "discriminators": discriminators.state_dict(),
                            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                            "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(), "step": step},
                           out / "last.pt")
            if step >= args.steps:
                break
    torch.save({"G": generator.state_dict(), "cfg": {**DECODER24, "sr": 24000, "hop": 960}, "step": step},
               out / "decoder24.pt")
    print("bitti:", out / "decoder24.pt")


if __name__ == "__main__":
    main()
