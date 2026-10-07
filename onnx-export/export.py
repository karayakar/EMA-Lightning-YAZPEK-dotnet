"""EMA Lightning'in uc model katmanini ONNX'e cevirir (tek seferlik).

Kurulum:     python -m pip install ema-lightning==1.0.1 onnx onnxruntime
Calistirma:  python export.py
Agirliklar:  ..\\ema-lightning\\ema.pt ve decoder.pt (HF kopyasi; hub'a gidilmez)
Cikti:       ..\\models\\text.onnx, sound.onnx, decoder.onnx, ema_meta.json
"""
import argparse
import inspect
import json
from pathlib import Path

import torch

from ema_lightning.decoder import load_decoder
from ema_lightning.engine import Engine
from ema_lightning.model import load_acoustic

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "ema-lightning"
OUT = ROOT / "models"
ACOUSTIC = WEIGHTS / "ema.pt"
DECODER = WEIGHTS / "decoder.pt"
OPSET = 17  # LayerNorm tek op, ScatterElements(reduction=add) >= 16


def _expm1(g, x):
    # ONNX'te expm1 yok: Exp(x) - 1 (Duration'da x <= 6, sonuç >= 1e-3'e kırpılıyor)
    return g.op("Sub", g.op("Exp", x), g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32)))


torch.onnx.register_custom_op_symbolic("aten::expm1", _expm1, OPSET)

# Iz cikarmak icin ornek girdi: model alfabesinde (kucuk harf, normalize edilmis) iki farkli uzunlukta parca,
# boylece batch ve padding birlikte izlenir
SAMPLE = ["merhaba, size nasıl yardımcı olabilirim?", "bugün hava çok güzel."]


class TextStage(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, mask):
        return self.model.text_stage(ids, mask)


class SoundStage(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, h, dur, mask, cw, wstart, fw, fp, fmask, noise):
        return self.model.sound_stage(h, dur, mask, cw, wstart, fw, fp, fmask, noise)


class DecodeStage(torch.nn.Module):
    # Engine.run ile ayni: z [B,T,64] gelir, decoder [B,64,T] ister; lengths her zaman verilir
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z, lengths):
        return self.decoder(z.transpose(1, 2), lengths)


def load():
    model = load_acoustic(str(ACOUSTIC), "cpu")
    decoder = load_decoder(str(DECODER), "cpu")
    return model, decoder, Engine(model, decoder, "cpu")


def stage_inputs(engine, texts, seed=0):
    """Engine.plan/think/decode'un modele verdigi tensorlerin aynisi."""
    pieces = [engine.piece(t, 0.0, seed + i) for i, t in enumerate(texts)]
    L = max(p.letters for p in pieces)
    ids = engine.stack([p.ids for p in pieces], L, 0)
    text = {"ids": ids, "mask": ids != 0}

    engine.plan(pieces, 1.0)
    T = max(p.frames for p in pieces)
    sound = {
        "h": engine.stack([p.h for p in pieces], L, 0.0),
        "dur": engine.stack([p.dur for p in pieces], L, 0.0),
        "mask": engine.stack([torch.ones(p.letters, dtype=torch.bool) for p in pieces], L, False),
        "cw": engine.stack([p.cw for p in pieces], L, -1),
        "wstart": engine.stack([p.wstart for p in pieces], L, 0),
        "fw": engine.stack([p.fw for p in pieces], T, -1),
        "fp": engine.stack([p.fp for p in pieces], T, 0.0),
        "fmask": engine.stack([torch.ones(p.frames, dtype=torch.bool) for p in pieces], T, False),
        "noise": engine.stack([engine.noise(p) for p in pieces], T, 0.0, dim=1),
    }

    engine.think(pieces)
    # Her parcanin tamami tek pencere; uzunluklar farkli oldugu icin padding maskesi de izlenir
    z = engine.stack([p.latents for p in pieces], T, 0.0)
    decode = {"z": z, "lengths": torch.tensor([p.frames for p in pieces], dtype=torch.long)}
    return text, sound, decode


def export(module, inputs, outputs, axes, path):
    kwargs = dict(input_names=list(inputs), output_names=outputs, dynamic_axes=axes,
                  opset_version=OPSET, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # yeni torch'larda varsayilan dynamo (onnxscript ister); TorchScript yolu
    torch.onnx.export(module, tuple(inputs.values()), str(path), **kwargs)
    print(f"yazildi: {path}")


def main():
    # Erkek seti: --acoustic ..\checkpoints\male\student\ema.pt --decoder ..\checkpoints\male\decoder\decoder24.pt
    #             --out ..\models\male
    global ACOUSTIC, DECODER, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic", default=str(ACOUSTIC))
    parser.add_argument("--decoder", default=str(DECODER))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    ACOUSTIC, DECODER, OUT = Path(args.acoustic), Path(args.decoder), Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    model, decoder, engine = load()
    text, sound, decode = stage_inputs(engine, SAMPLE)

    BL, BT = {0: "B", 1: "L"}, {0: "B", 1: "T"}
    with torch.no_grad():
        export(TextStage(model), text, ["h", "dur"],
               {"ids": BL, "mask": BL, "h": BL, "dur": BL}, OUT / "text.onnx")
        export(SoundStage(model), sound, ["latents"],
               {"h": BL, "dur": BL, "mask": BL, "cw": BL, "wstart": BL,
                "fw": BT, "fp": BT, "fmask": BT, "noise": {0: "B", 2: "T"}, "latents": BT},
               OUT / "sound.onnx")
        export(DecodeStage(decoder), decode, ["audio"],
               {"z": BT, "lengths": {0: "B"}, "audio": {0: "B", 1: "S"}}, OUT / "decoder.onnx")

    # C# tarafinin ihtiyac duydugu model bilgileri
    meta = {
        "vocab": model.vocab,
        "times": model.times,
        "latent_dim": model.latent_dim,
        "d": model.d,
        "hop": decoder.hop,
        "sample_rate": decoder.hop * 25,  # 25 Hz latent: hop 1920 → 48 kHz, hop 960 → 24 kHz
        "latent_rate": 25,
        "opset": OPSET,
    }
    (OUT / "ema_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"yazildi: {OUT / 'ema_meta.json'}")


if __name__ == "__main__":
    main()
