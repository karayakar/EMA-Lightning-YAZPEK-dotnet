"""T2a — CTC hizalayıcı: yalnız kullanıcının verisiyle (dış ASR modeli yok). Model alfabesi (49) + blank.

  .venv\\Scripts\\python.exe t2_ctc.py
Çıktı: ..\\checkpoints\\male\\ctc\\ (last.pt, ctc.pt, tb\\)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import load_manifest, load_segment  # noqa: E402
from common.mel import Mel  # noqa: E402
from common.nets import ConvNeXt1d  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLANK = 0  # vocab[0] = '<pad>' metinde hiç geçmez; CTC blank olarak kullanılır


class CtcModel(nn.Module):
    """log-mel (100 kare/sn) → 50 kare/sn harf olasılıkları."""

    def __init__(self, vocab_size, n_mels=100, ch=256, blocks=6):
        super().__init__()
        self.pre = nn.Conv1d(n_mels, ch, 5, padding=2)
        self.down = nn.Conv1d(ch, ch, 4, stride=2, padding=1)
        self.blocks = nn.Sequential(*(ConvNeXt1d(ch) for _ in range(blocks)))
        self.rnn = nn.LSTM(ch, ch, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * ch, vocab_size)

    def forward(self, mel):
        x = self.blocks(self.down(F.gelu(self.pre(mel))))
        x, _ = self.rnn(x.transpose(1, 2))
        return self.head(x)  # [B, T/2, V]


def frames50(samples):
    return samples // 240 // 2


class Utterances(torch.utils.data.Dataset):
    def __init__(self, records, stoi):
        self.records = records
        self.stoi = stoi

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        audio = load_segment(r)
        audio = audio[: len(audio) // 960 * 960]
        labels = [self.stoi.get(c, 1) for c in r["text_model"]]
        return torch.from_numpy(np.ascontiguousarray(audio)), torch.tensor(labels), index


def collate(items):
    audio = torch.nn.utils.rnn.pad_sequence([a for a, _, _ in items], batch_first=True)
    lengths = torch.tensor([len(a) for a, _, _ in items])
    labels = torch.cat([l for _, l, _ in items])
    label_lengths = torch.tensor([len(l) for _, l, _ in items])
    return audio, lengths, labels, label_lengths, [i for _, _, i in items]


def length_batches(records, max_seconds, seed):
    """Benzer uzunluktakileri bir araya getiren, toplam süresi max_seconds'u aşmayan batch'ler."""
    order = sorted(range(len(records)), key=lambda i: records[i]["seconds"])
    batches, current, total = [], [], 0.0
    for i in order:
        s = records[i]["seconds"]
        if current and total + s > max_seconds:
            batches.append(current)
            current, total = [], 0.0
        current.append(i)
        total += s
    if current:
        batches.append(current)
    random.Random(seed).shuffle(batches)
    return batches


def greedy(log_probs, length):
    ids = log_probs[:length].argmax(-1).tolist()
    out, previous = [], None
    for i in ids:
        if i != previous and i != BLANK:
            out.append(i)
        previous = i
    return out


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data" / "male" / "manifest.jsonl"))
    parser.add_argument("--weights", default=str(ROOT / "ema-lightning" / "ema.pt"))
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "male" / "ctc"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-seconds", type=float, default=240.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--init", default=None, help="başlangıç CTC ağırlıkları (ctc.pt)")
    args = parser.parse_args()

    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    vocab = torch.load(args.weights, map_location="cpu", weights_only=False)["vocab"]
    stoi = {c: i for i, c in enumerate(vocab)}
    train_records = [r for r in load_manifest(args.manifest, "train") if r["seconds"] <= 20]
    val_records = load_manifest(args.manifest, "val")[:300]

    mel = Mel().to(device)
    model = CtcModel(len(vocab)).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = len(length_batches(train_records, args.batch_seconds, 0))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.epochs * steps_per_epoch,
                                                    pct_start=0.1)
    start_epoch, step = 0, 0
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, step = state["epoch"], state["step"]
        print(f"devam: epoch {start_epoch}")
    writer = SummaryWriter(out / "tb")
    train_set = Utterances(train_records, stoi)
    val_loader = torch.utils.data.DataLoader(Utterances(val_records, stoi), batch_size=16, collate_fn=collate)

    @torch.no_grad()
    def evaluate():
        model.eval()
        errors = chars = 0
        for audio, lengths, labels, label_lengths, _ in val_loader:
            log_probs = model(mel(audio.to(device))).float().log_softmax(-1).cpu()
            offset = 0
            for b in range(audio.shape[0]):
                target = labels[offset:offset + label_lengths[b]].tolist()
                offset += label_lengths[b]
                errors += edit_distance(greedy(log_probs[b], frames50(lengths[b].item())), target)
                chars += len(target)
        model.train()
        return errors / max(chars, 1)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        loader = torch.utils.data.DataLoader(train_set, batch_sampler=length_batches(train_records, args.batch_seconds,
                                                                                     epoch),
                                             collate_fn=collate, num_workers=args.workers, pin_memory=True)
        started = time.time()
        for n, (audio, lengths, labels, label_lengths, _) in enumerate(loader, 1):
            audio = audio.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(mel(audio))
            log_probs = logits.float().log_softmax(-1).transpose(0, 1)
            input_lengths = torch.clamp(frames50(lengths), max=log_probs.shape[0])
            loss = F.ctc_loss(log_probs, labels.to(device), input_lengths.to(device), label_lengths.to(device),
                              blank=BLANK, zero_infinity=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if n % 100 == 0:
                writer.add_scalar("train/ctc", loss.item(), step)
                rate = n / (time.time() - started)
                print(f"epoch {epoch + 1}/{args.epochs} batch {n}/{steps_per_epoch} ctc {loss.item():.3f} "
                      f"{rate:.1f} batch/sn  epoch kalan ~{(steps_per_epoch - n) / rate / 60:.0f} dk", flush=True)
        cer = evaluate()
        writer.add_scalar("val/cer", cer, step)
        print(f"epoch {epoch + 1} bitti: val CER {cer:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "step": step}, out / "last.pt")
    torch.save({"model": model.state_dict(), "vocab": vocab, "val_cer": evaluate()}, out / "ctc.pt")
    print("bitti:", out / "ctc.pt")


if __name__ == "__main__":
    main()
