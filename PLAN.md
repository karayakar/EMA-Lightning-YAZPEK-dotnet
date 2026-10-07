# EMA Lightning → ONNX → .NET 8 (CPU) — Plan

Kaynak: `github\ema-lightning` (Python paketi 1.0.1), `ema-lightning\` (HF ağırlıkları),
`github\normalizer-tr` (Rust 0.4.0). Orijinal repolara dokunulmaz.

## Kararlar (2026-10-07, kullanıcı onaylı)

| Konu | Karar |
|---|---|
| Hedef | .NET 8, CPU, ONNX Runtime |
| Normalizer | normalizer-tr C#'a **tam port** (tüm public API, 3 politika, hint, segment, issue, fallback, limitler) |
| Servis | ASP.NET Core: HTTP (tam WAV) + WebSocket (stream) |
| Çoklu istek | Playhead birebir port: iki FIFO kuyruk, tek arka plan döngüsü, istek başına outbox, kopan istek düşer, hata sadece kendi batch'ine |
| Seed | .NET kendi RNG'si; Python ile aynı ses beklenmez, .NET içinde aynı seed aynı ses |
| Batch size | Python'daki gibi: açılışta 1/2/4/8 ölç, %90 kuralı, sonucu cache dosyasına yaz |
| Stream formatı | float32 PCM mono, sample_rate 48000/24000/16000/8000 |
| Paketler | Python: `onnx`, `onnxruntime` (+ `ema-lightning`). .NET: `Microsoft.ML.OnnxRuntime`. Test projesi YOK |
| Timeline (`plan`) | Repodaki gibi host (C#) tarafında; ONNX'te sadece 3 model katmanı |

## Klasörler

```
onnx-export\        export.py, verify.py
models\             text.onnx, sound.onnx, decoder.onnx, ema_meta.json   (export.py üretir)
dotnet\
  EmaLightning.Normalizer\   normalizer-tr C# portu (namespace NormalizerTr)
  EmaLightning.Normalizer.Parity\  Rust beklenen çıktılarıyla karşılaştırma konsolu
  EmaLightning.Core\         Frontend, Chunker, Engine, Playhead, Resampler, Wav
  EmaLightning.Server\       ASP.NET Core HTTP + WebSocket
```

## ONNX katmanları

| Dosya | Kaynak | Giriş | Çıkış |
|---|---|---|---|
| `text.onnx` | `Acoustic.text_stage` | `ids` int64 [B,L], `mask` bool [B,L] | `h` f32 [B,L,224], `dur` f32 [B,L] |
| `sound.onnx` | `Acoustic.sound_stage` | `h`, `dur`, `mask`, `cw` i64, `wstart` i64, `fw` i64 [B,T], `fp` f32 [B,T], `fmask` bool [B,T], `noise` f32 [B,4,T,64] | `latents` f32 [B,T,64] |
| `decoder.onnx` | `Decoder.forward` | `z` f32 [B,T,64], `lengths` i64 [B] | `audio` f32 [B,T*1920] |

## Fazlar

1. **ONNX export + doğrulama** (Python) — `onnx-export\`
2. **Normalizer C# portu** — Rust test fixture'larıyla (`tests\fixtures\*`) parity, konsol aracıyla
3. **Core** — Frontend, Chunker, Engine, Playhead, Resampler, Wav, batch size ölçümü
4. **Server** — HTTP + WebSocket uçları

## Servis (Faz 4 kararları)

- `POST /tts` JSON `{text, speed?, seed?, sample_rate?}` → `audio/wav`
- `WS /speak` ilk mesaj aynı JSON → float32 PCM LE mono binary parçalar, bitince Normal close
- appsettings.json: `Ema:ModelDirectory` (varsayılan `..\..\models`), Kestrel `http://localhost:5180`
- Açılışta `BestBatchSize()`; CORS açık, auth yok

Build ve çalıştırma kullanıcıda.
