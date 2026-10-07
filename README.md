# EMA Lightning YAPZEK — .NET

Turkish text-to-speech based on EMA Lightning: ONNX export, a .NET 8 CPU inference service and a training pipeline for new speakers.

## Original projects

This work is built on the following projects:

- **EMA Lightning** — https://github.com/canberk7/ema-lightning
  Turkish TTS model (text encoder, duration, aligner, DiT, DMD2 4-step sampler, HiFi-GAN decoder).
- **normalizer-tr** — https://github.com/erdemtuna/normalizer-tr
  Turkish text normalizer; ported to C# in `dotnet/EmaLightning.Normalizer`.

## Contents

| Folder | Description |
|---|---|
| `onnx-export/` | Exports the model to ONNX (`text.onnx`, `sound.onnx`, `decoder.onnx`, `ema_meta.json`) and verifies against PyTorch |
| `dotnet/EmaLightning.Normalizer/` | C# port of normalizer-tr |
| `dotnet/EmaLightning.Core/` | ONNX Runtime inference: frontend, chunker, engine, streaming playhead, expression clips |
| `dotnet/EmaLightning.Server/` | ASP.NET Core service: `POST /tts`, `WS /speak`, `GET /voices`, test page |
| `native/world_shift/` | Pitch / formant shifter (WORLD) |
| `training/` | Training pipeline: data preparation, encoder, 24 kHz decoder, CTC aligner, flow-matching teacher, DMD2 distillation, evaluation |

## Speakers

Each speaker is a folder under `models/` with `text.onnx`, `sound.onnx`, `decoder.onnx` and `ema_meta.json`. The server reads them from `Ema:Speakers` in `dotnet/EmaLightning.Server/appsettings.json`.

| Speaker | Folder | Sample rate | Expression clips |
|---|---|---|---|
| `female` | `models/` | 48 kHz | — |
| `male` | `models/male/` | 24 kHz | — |
| `beyza` | `models/beyza/` | 24 kHz | `models/beyza/expressions/` |
| `beyzaR` | `models/beyzaR/` | 24 kHz | `models/beyzaR/expressions/` |

## Expressions

Speakers with an `expressions/` folder play recorded clips for tags and emojis in the text. A tag picks a random clip of that category; a numbered tag (`<laugh_03>`) picks that exact clip. Unknown tags and unmapped emojis are skipped silently.

Example: `Buna çok güldüm 😂 ama sonra üzüldüm 😢` or `Bunu duyunca <laugh> gerçekten şaşırdım.`

### Emojis

| Emoji | Tag |
|---|---|
| 😂 🤣 😆 | `<laugh>` |
| 😢 😭 | `<cry>` |
| 😔 | `<sad>` |
| 😮‍💨 😌 | `<sigh>` |
| 🥱 | `<yawn>` |
| 🤧 | `<sniff>` |
| 😷 | `<cough>` |
| 😩 | `<ufff>` |
| 😫 | `<off>` |
| 🙄 | `<puff>` |
| 😱 😨 | `<eyvah>` |
| 😬 | `<clear_throat>` |

### Tags

| Tag | Clips |
|---|---|
| `<laugh>` | 01–08 |
| `<cough>` | 01–09 |
| `<sigh>` | 01–06 |
| `<sniff>` | 01–05 |
| `<clear_throat>` | 01–04 |
| `<cry>` | 01, 02 |
| `<ayy>` | 01, 02, 04 |
| `<puff>` | 01, 03 |
| `<sad>`, `<yawn>`, `<eyvah>`, `<ufff>`, `<off>`, `<offf>`, `<off_off>`, `<oof_off>`, `<pufff>`, `<allah>`, `<ay_ay>`, `<oyy_oyy>`, `<aiiyyy>`, `<aaayyyyi>`, `<iiiyyy>`, `<aayyy_off>`, `<I_Im>` | 01 |
| `<ayy_ayy_yorgun>` | 01 |
| `<ayy_ayyy>`, `<ayyy_ayy>` | 03 |
| `<ah>` | 1 |

Datasets and training checkpoints are not included in this repository.

## License

Apache License 2.0 — see [LICENSE](LICENSE). The original projects (ema-lightning, normalizer-tr) are also licensed under Apache 2.0.
