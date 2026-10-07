"""T3 öncesi: T1a encoder'ı ile tüm kayıtların latentleri (25 Hz × 64, float16) tek dosyaya.

  .venv\\Scripts\\python.exe t3_latents.py
Çıktı: ..\\data\\male\\latents.pt  {id: Tensor[T, 64] float16}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import load_manifest, load_segment  # noqa: E402
from common.mel import Mel  # noqa: E402
from common.nets import Encoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class Segments(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        audio = load_segment(self.records[index])
        return torch.from_numpy(np.ascontiguousarray(audio[: len(audio) // 960 * 960])), index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data" / "male" / "manifest.jsonl"))
    parser.add_argument("--encoder", default=str(ROOT / "checkpoints" / "male" / "encoder" / "encoder.pt"))
    parser.add_argument("--out", default=str(ROOT / "data" / "male" / "latents.pt"))
    args = parser.parse_args()

    device = torch.device("cuda")
    encoder = Encoder().to(device).eval()
    encoder.load_state_dict(torch.load(args.encoder, map_location=device)["encoder"])
    mel = Mel().to(device)
    records = load_manifest(args.manifest)
    loader = torch.utils.data.DataLoader(Segments(records), batch_size=None, num_workers=6)
    latents = {}
    with torch.no_grad():
        for n, (audio, index) in enumerate(loader, 1):
            if audio.numel() == 0:
                continue
            z = encoder(mel(audio.to(device)[None]))[0].transpose(0, 1)
            latents[records[index]["id"]] = z.half().cpu()
            if n % 10000 == 0:
                print(f"  {n}/{len(records)}", flush=True)
    torch.save(latents, args.out)
    frames = sum(z.shape[0] for z in latents.values())
    print(f"bitti: {len(latents)} kayıt, {frames} kare → {args.out}")


if __name__ == "__main__":
    main()
