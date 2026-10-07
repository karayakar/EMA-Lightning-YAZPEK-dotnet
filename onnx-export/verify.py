"""PyTorch ile ONNX Runtime (CPU) ciktilarini ayni girdi ve ayni gurultuyle karsilastirir.

Calistirma:  python verify.py   (once export.py)
Export'takinden farkli batch boyutu ve uzunluklar kullanilir; dinamik eksenler de sinanmis olur.
"""
import numpy as np
import onnxruntime as ort
import torch

from export import OUT, TextStage, SoundStage, DecodeStage, load, stage_inputs

TEXTS = [
    "sabahın erken saatlerinde liman henüz uyanmamıştı.",
    "günaydın.",
    "siparişiniz yola çıktı, yarın öğleden sonra elinizde olacak.",
]


def session(name):
    return ort.InferenceSession(str(OUT / name), providers=["CPUExecutionProvider"])


def run_ort(sess, inputs):
    return sess.run(None, {k: v.numpy() for k, v in inputs.items()})


def report(name, ref, got):
    ref = ref.detach().numpy()
    diff = np.abs(ref - got)
    print(f"{name:10s} shape={tuple(got.shape)}  max_abs={diff.max():.3e}  mean_abs={diff.mean():.3e}")


def main():
    model, decoder, engine = load()
    text, sound, decode = stage_inputs(engine, TEXTS, seed=7)

    with torch.no_grad():
        h, dur = TextStage(model)(**text)
        latents = SoundStage(model)(**sound)
        audio = DecodeStage(decoder)(**decode)

    oh, odur = run_ort(session("text.onnx"), text)
    (olat,) = run_ort(session("sound.onnx"), sound)
    (oaudio,) = run_ort(session("decoder.onnx"), decode)

    print(f"onnxruntime {ort.__version__}, torch {torch.__version__}")
    report("h", h, oh)
    report("dur", dur, odur)
    report("latents", latents, olat)
    report("audio", audio, oaudio)


if __name__ == "__main__":
    main()
