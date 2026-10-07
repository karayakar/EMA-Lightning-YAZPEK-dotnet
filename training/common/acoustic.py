"""T3/T4 ortak: eğitim örnekleri (metin + kelime süreleri + latent), koşul hesabı, örnekleme, zaman çizelgesi.

Akustik model ema_lightning.model.Acoustic (checkpoint config'i ile, text_attn = 0) — mimari değişmez.
"""
import json
import random

import numpy as np
import torch

from ema_lightning.model import rope

MAX_WORD_FRAMES = 250
MAX_FRAMES = 3000


def piece_arrays(text, stoi):
    """engine.piece ile birebir: harf id'leri, her harfin kelimesi (cw), kelimenin ilk harfi (wstart)."""
    ids = [stoi.get(ch, 1) for ch in text]
    starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
    bounds = [0] + starts[1:] + [len(text)]
    cw, wstart = [], []
    for w in range(len(bounds) - 1):
        cw += [w] * (bounds[w + 1] - bounds[w])
        wstart += [bounds[w]] * (bounds[w + 1] - bounds[w])
    return ids, cw, wstart


def timeline(counts):
    """Kelime kare sayıları → her karenin kelimesi (fw) ve kelime içi konumu (fp) — engine.plan ile aynı formül."""
    n = np.asarray(counts, np.int64)
    fw = np.repeat(np.arange(len(n)), n)
    before = np.cumsum(n) - n
    fp = ((np.arange(len(fw)) - before[fw]) / n[fw]).astype(np.float32)
    return fw, fp


class Examples(torch.utils.data.Dataset):
    """(metin, kelime süreleri, latent) örnekleri; latents: {id: float16 [T, 64]}."""

    def __init__(self, manifest_records, durations, latents, stoi, max_frames=250, min_score=float("-inf")):
        self.items = []
        for r in manifest_records:
            d = durations.get(r["id"])
            z = latents.get(r["id"])
            if d is None or z is None or d["frames"] != z.shape[0] or d["frames"] > max_frames:
                continue
            if max(d["counts"]) > MAX_WORD_FRAMES or d["score"] < min_score:
                continue  # düşük hizalama skoru: metin/ses uyuşmazlığı (T2 raporu)
            self.items.append((r["text_model"], d["counts"], r["id"]))
        self.latents = latents
        self.stoi = stoi

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        text, counts, uid = self.items[index]
        ids, cw, wstart = piece_arrays(text, self.stoi)
        fw, fp = timeline(counts)
        return {"ids": ids, "cw": cw, "wstart": wstart, "counts": counts, "fw": fw, "fp": fp,
                "z": self.latents[uid], "id": uid}


def collate(items):
    """engine.think ile aynı doldurma değerleri: ids 0, cw -1, wstart 0, fw -1, fp 0, maskeler False."""
    b = len(items)
    L = max(len(x["ids"]) for x in items)
    T = max(len(x["fw"]) for x in items)
    W = max(len(x["counts"]) for x in items)
    batch = {
        "ids": torch.zeros(b, L, dtype=torch.long), "mask": torch.zeros(b, L, dtype=torch.bool),
        "cw": torch.full((b, L), -1, dtype=torch.long), "wstart": torch.zeros(b, L, dtype=torch.long),
        "fw": torch.full((b, T), -1, dtype=torch.long), "fp": torch.zeros(b, T),
        "fmask": torch.zeros(b, T, dtype=torch.bool), "z": torch.zeros(b, T, 64),
        "counts": torch.zeros(b, W), "wmask": torch.zeros(b, W, dtype=torch.bool), "id": [x["id"] for x in items],
    }
    for i, x in enumerate(items):
        n, t, w = len(x["ids"]), len(x["fw"]), len(x["counts"])
        batch["ids"][i, :n] = torch.tensor(x["ids"])
        batch["mask"][i, :n] = True
        batch["cw"][i, :n] = torch.tensor(x["cw"])
        batch["wstart"][i, :n] = torch.tensor(x["wstart"])
        batch["fw"][i, :t] = torch.from_numpy(x["fw"])
        batch["fp"][i, :t] = torch.from_numpy(x["fp"])
        batch["fmask"][i, :t] = True
        batch["z"][i, :t] = torch.as_tensor(np.asarray(x["z"], np.float32))
        batch["counts"][i, :w] = torch.tensor(x["counts"], dtype=torch.float32)
        batch["wmask"][i, :w] = True
    return batch


def frame_batches(lengths, max_frames, seed):
    """Benzer uzunlukları gruplayan, toplam karesi max_frames'i aşmayan batch'ler (karışık sıra)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, current, longest = [], [], 0
    for i in order:
        longest_next = max(longest, lengths[i])
        if current and longest_next * (len(current) + 1) > max_frames:
            batches.append(current)
            current, longest_next = [], lengths[i]
        current.append(i)
        longest = longest_next
    if current:
        batches.append(current)
    random.Random(seed).shuffle(batches)
    return batches


def condition(model, h, batch, dur):
    """sound_stage'in koşul kısmı: harf konumları (dur'dan) + pencereli aligner → c [B, T, d]."""
    mask, cw, wstart = batch["mask"], batch["cw"], batch["wstart"]
    c = dur.clamp(min=1e-4) * mask
    done = c.cumsum(-1)
    before = done - c
    word = cw.clamp(min=0)
    total = torch.zeros_like(c).scatter_add_(1, word, c).gather(1, word)
    cp = ((done - before.gather(1, wstart) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * mask
    return model.aligner(h, cw, cp, batch["fw"], batch["fp"], mask, cw.shape[1])


def word_sums(dur, cw, words):
    return torch.zeros(dur.shape[0], words, device=dur.device, dtype=dur.dtype).scatter_add_(1, cw.clamp(min=0),
                                                                                           dur)


def rotary(model, frames, device):
    return rope(torch.arange(frames, device=device), model.dh)


@torch.no_grad()
def sample_teacher(model, cond, fmask, steps=16, guidance=4.0, generator=None):
    """Orta nokta ODE (t: 0 → 1), CFG: v = v_u + g (v_c − v_u). x_t = (1−t)ε + t·z."""
    b, t_len, _ = cond.shape
    cos, sin = rotary(model, t_len, cond.device)
    x = torch.randn(b, t_len, model.latent_dim, device=cond.device, generator=generator)
    zero = torch.zeros_like(cond)

    def velocity(x, t):
        tt = torch.full((b,), t, device=cond.device)
        vc = model.backbone(x, cond, tt, fmask, cos, sin)
        vu = model.backbone(x, zero, tt, fmask, cos, sin)
        return vu + guidance * (vc - vu)

    h = 1.0 / steps
    for k in range(steps):
        t = k * h
        mid = x + 0.5 * h * velocity(x, t)
        x = x + h * velocity(mid, t + 0.5 * h)
    return x


def predicted_timeline(dur, cw, mask, speed=1.0):
    """engine.plan: tahmini harf süreleri → kelime kare sayıları (yuvarla, 1..250) → fw, fp (batch, doldurulmuş)."""
    dur = dur / speed
    words = int(cw.max().item()) + 1
    sums = word_sums(dur * mask, cw, words).round().clamp(1, MAX_WORD_FRAMES).long().cpu()
    fws, fps = [], []
    for i in range(dur.shape[0]):
        n = sums[i, : int(cw[i][mask[i]].max().item()) + 1].numpy()
        fw, fp = timeline(n[: max(1, len(n))])
        fws.append(fw[:MAX_FRAMES])
        fps.append(fp[:MAX_FRAMES])
    T = max(len(f) for f in fws)
    fw_t = torch.full((len(fws), T), -1, dtype=torch.long)
    fp_t = torch.zeros(len(fws), T)
    fmask = torch.zeros(len(fws), T, dtype=torch.bool)
    for i, (fw, fp) in enumerate(zip(fws, fps)):
        fw_t[i, : len(fw)] = torch.from_numpy(fw)
        fp_t[i, : len(fp)] = torch.from_numpy(fp)
        fmask[i, : len(fw)] = True
    return fw_t.to(dur.device), fp_t.to(dur.device), fmask.to(dur.device)


def load_durations(path):
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in map(json.loads, f)}
