"""T1a — latent encoder: orijinal 48 kHz decoder dondurulmuş, loss 24 kHz'te (TRAINING_PROPOSAL T1a).

  .venv\\Scripts\\python.exe t1_encoder.py            (kaldığı yerden devam eder)
Çıktı: ..\\checkpoints\\male\\encoder\\  (last.pt, latent_stats.pt, samples\\, tb\\)
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
from common.audio import read_wav  # noqa: E402
from common.data import Clips, load_manifest, write_samples  # noqa: E402
from common.mel import Downsample2, Mel, MultiResolutionStft  # noqa: E402
from common.nets import Encoder, frozen_decoder48  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def latent_stats(manifest, weights, decoder, device, count, out):
    """Mevcut (kadın) akustik modelin ürettiği latentlerin boyut başına ortalama/std'si — encoder uzay uyumu için."""
    from ema_lightning.engine import Engine
    from ema_lightning.model import load_acoustic

    model = load_acoustic(str(weights), device)
    engine = Engine(model, decoder, device)
    texts = [r["text_model"] for r in load_manifest(manifest, "train") if not r["long"]][:count]
    latents = []
    for i in range(0, len(texts), 32):
        pieces = [engine.piece(t, 0.0, i + j) for j, t in enumerate(texts[i:i + 32])]
        engine.plan(pieces, 1.0)
        engine.think(pieces)
        latents += [p.latents.float() for p in pieces]
    z = torch.cat(latents)
    stats = {"mean": z.mean(0).cpu(), "std": z.std(0).cpu(), "frames": z.shape[0]}
    torch.save(stats, out)
    print(f"latent istatistiği: {len(texts)} cümle, {z.shape[0]} kare, ort. |mean| {stats['mean'].abs().mean():.3f}, "
          f"ort. std {stats['std'].mean():.3f}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data" / "male" / "manifest.jsonl"))
    parser.add_argument("--weights", default=str(ROOT / "ema-lightning"))
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "male" / "encoder"))
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=2.88)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--moment", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--every", type=int, default=2500, help="örnek + checkpoint aralığı (adım)")
    parser.add_argument("--init", default=None, help="başlangıç encoder ağırlıkları (encoder.pt)")
    args = parser.parse_args()

    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    decoder = frozen_decoder48(Path(args.weights) / "decoder.pt", device)
    stats_path = out / "latent_stats.pt"
    stats = torch.load(stats_path) if stats_path.exists() else latent_stats(
        args.manifest, Path(args.weights) / "ema.pt", decoder, device, 2000, stats_path)
    ref_mean, ref_std = stats["mean"].to(device), stats["std"].to(device)

    mel, down, mrstft = Mel().to(device), Downsample2().to(device), MultiResolutionStft().to(device)
    encoder = Encoder().to(device)
    if args.init:
        encoder.load_state_dict(torch.load(args.init, map_location=device)["encoder"])
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * 0.05)
    step = 0
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device)
        encoder.load_state_dict(state["encoder"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        print(f"devam: adım {step}")

    train = torch.utils.data.DataLoader(Clips(args.manifest, "train", args.seconds), batch_size=args.batch,
                                        shuffle=True, num_workers=args.workers, drop_last=True,
                                        persistent_workers=True, pin_memory=True)
    val = next(iter(torch.utils.data.DataLoader(Clips(args.manifest, "val", 4.8, seed=1), batch_size=6)))
    writer = SummaryWriter(out / "tb")

    def reconstruct(x):
        z = encoder(mel(x))
        return down(decoder(z))[:, : x.shape[1]], z

    def losses(x):
        y, z = reconstruct(x)
        mel_loss = F.l1_loss(mel(y), mel(x))
        stft_loss = mrstft(y, x)
        zm, zs = z.mean(dim=(0, 2)), z.std(dim=(0, 2))
        moment = F.mse_loss(zm, ref_mean) + F.mse_loss(zs, ref_std)
        return mel_loss, stft_loss, moment

    @torch.no_grad()
    def evaluate():
        encoder.eval()
        x = val.to(device)
        mel_loss, stft_loss, moment = losses(x)
        writer.add_scalar("val/mel", mel_loss.item(), step)
        writer.add_scalar("val/stft", stft_loss.item(), step)
        samples = out / "samples" / f"{step:06d}"
        y, _ = reconstruct(x)
        for i in range(x.shape[0]):
            write_samples(samples, f"val{i}_orig", x[i].cpu().numpy())
            write_samples(samples, f"val{i}_recon", y[i].float().cpu().numpy())
        # kadın referansı (48 kHz assets → 24 kHz): encoder'ın kaynak uzayı koruyup korumadığı
        for k in (1, 2, 3):
            ref, _, _ = read_wav(Path(args.weights) / "assets" / f"sample-{k}.wav")
            r = down(torch.from_numpy(ref).to(device)[None])
            r = r[:, : r.shape[1] // 960 * 960]
            ry, _ = reconstruct(r)
            write_samples(samples, f"female{k}_recon", ry[0].float().cpu().numpy())
        encoder.train()
        print(f"  val adım {step}: mel {mel_loss.item():.4f} stft {stft_loss.item():.4f} moment {moment.item():.4f}"
              f" → {samples}", flush=True)

    encoder.train()
    started, seen = time.time(), 0
    while step < args.steps:
        for x in train:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mel_loss, stft_loss, moment = losses(x)
                loss = mel_loss + stft_loss + args.moment * moment
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            seen += 1
            if step % 100 == 0:
                rate = seen / (time.time() - started)
                writer.add_scalar("train/mel", mel_loss.item(), step)
                writer.add_scalar("train/stft", stft_loss.item(), step)
                writer.add_scalar("train/moment", moment.item(), step)
                print(f"adım {step}/{args.steps}  mel {mel_loss.item():.4f}  stft {stft_loss.item():.4f}  "
                      f"moment {moment.item():.4f}  {rate:.1f} adım/sn  "
                      f"kalan ~{(args.steps - step) / rate / 60:.0f} dk", flush=True)
            if step % args.every == 0 or step == args.steps:
                evaluate()
                torch.save({"encoder": encoder.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "step": step}, out / "last.pt")
            if step >= args.steps:
                break
    torch.save({"encoder": encoder.state_dict(), "step": step}, out / "encoder.pt")
    print("bitti:", out / "encoder.pt")


if __name__ == "__main__":
    main()
