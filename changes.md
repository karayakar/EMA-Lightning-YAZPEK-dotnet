# changes.md

## 2026-10-07 — Faz 1: ONNX export

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| Kararların ve faz planının kaydı | `PLAN.md` | yok | yeni dosya |
| 3 model katmanını (text/sound/decoder) ONNX'e çevirmek; C# için `ema_meta.json` (vocab, times, hop…) | `onnx-export\export.py` | yok | yeni dosya |
| PyTorch ↔ ONNX Runtime CPU çıktı farkını ölçmek (farklı batch/uzunlukla) | `onnx-export\verify.py` | yok | yeni dosya |

## 2026-10-07 — Faz 1 düzeltme: expm1 export

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| `aten::expm1` opset 17'de yok → export hatası; kullanıcı 1. yolu seçti (yedek: `20261007_025922_export.py.bck`) | `onnx-export\export.py` | `OPSET = 17` sonrası boş | `_expm1(g, x)` = `Sub(Exp(x), 1)` + `torch.onnx.register_custom_op_symbolic("aten::expm1", _expm1, OPSET)` |
| Python 3.12 venv + torch 2.14.1+cpu, ema-lightning 1.0.1, onnx 1.23.2, onnxruntime 1.30.0 (kullanıcı onaylı) | `onnx-export\.venv\` | yok | kuruldu |

## 2026-10-07 — Ses dönüştürücü (B yolu: YIN + TD-PSOLA + LPC zarf bükme), kullanıcı isteği

Yedekler: `20261007_031249_Ema.cs.bck`, `20261007_031249_Program.cs.bck`, `wwwroot\20261007_031249_index.html.bck`

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| F0 (PSOLA) ve formant (LPC/STFT) dönüştürme, 48 kHz akış halinde | `dotnet\EmaLightning.Core\VoiceShifter.cs` | yok | yeni |
| `pitch`/`formant` parametreleri (0.5–2 / 0.7–1.4, varsayılan 1 = değişiklik yok); Decode → VoiceShifter → Resampler | `dotnet\EmaLightning.Core\Ema.cs` | SayAsync/SayManyAsync/StreamAsync/Collect/Receive/Check pitch-formant almıyordu | alıyor; `Receive` içinde `VoiceShifter.Create` + Push/Flush |
| İstek JSON'una `pitch`, `formant` | `dotnet\EmaLightning.Server\Program.cs` | TtsRequest(text, speed, seed, sample_rate) | + pitch, formant |
| Voice seçimi: Female (original) 1/1, Male 0.5/0.86, Custom + Pitch/Formant kutuları | `dotnet\EmaLightning.Server\wwwroot\index.html` | yok | eklendi |

## 2026-10-07 — ASR önerisi (PARK)

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| Türkçe ASR: Conformer-CTC (akış + offline), konuşmacı bazlı test, ITN, opsiyonel n-gram LM, ONNX + .NET; fazlar A0–A5; kod yok | `ASR_PROPOSAL.md` | yok | yeni |

## 2026-10-07 — Çok konuşmacılı model önerisi (PARK)

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| Konuşmacı kimliği (speaker embedding) mimarisi, fazlar M0–M5, açık sorular; kod yok | `MULTI_SPEAKER_PROPOSAL.md` | yok | yeni |

## 2026-10-07 — Stream ilk ses hızlandırma (kullanıcı isteği: "ilk kelime ile başlasın")

Yedekler: `dotnet\EmaLightning.Core\20261007_125930_{Engine,Playhead,Ema,Chunker}.cs.bck`
Ölçüm (13.8 sn metin, erkek): ilk parça 1.493 ms → 227–276 ms; toplam ~2.1–2.2 sn (değişmedi).

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| Parça tek başına düşünülsün işareti | `Engine.cs` (Piece) | yok | `Solo` |
| Solo parça batch'e başkasıyla girmez; batch Solo'da kesilir | `Playhead.cs` (Think) | ilk 8 cümle tek batch | Solo → batch 1 |
| Yalnız stream: ilk kelime ayrı parça + ilk cümlenin kalanı ayrı (ikisi Solo, aralarında duraklama yok) | `Ema.cs` (Pieces, Submit, StreamCore) | chunker parçaları aynen | ilk parça ikiye bölünür |
| Finish erişimi | `Chunker.cs` | `static string Finish` | `internal static string Finish` |

## 2026-10-07 gece — Erkek ses eğitimi (T0–T4 kodu) ve T5 hazırlığı (kullanıcı: "kontrol sende, soru sorma")

Ayrıntılı akış ve kararlar: `TRAINING_STATUS.md`.
Yedekler: `onnx-export\20261007_*_export.py.bck`, `dotnet\*\20261007_042535_*.bck` (Engine, Ema, Audio, Playhead, Program, appsettings, index.html)

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| Eğitim kodu (yeni) | `training\t0_prepare.py`, `t1_encoder.py`, `t1_decoder.py`, `t2_ctc.py`, `t2_align.py`, `t3_latents.py`, `t3_teacher.py`, `t4_distill.py`, `common\*.py` | yok | yeni |
| Erkek seti dışa aktarımı | `onnx-export\export.py` | sabit `ema-lightning\` ağırlıkları, `sample_rate` 48000 | `--acoustic/--decoder/--out`; `sample_rate = hop × 25` |
| Model örnekleme hızı meta'dan | `dotnet\EmaLightning.Core\Engine.cs` | yalnız `const Rate = 48000` | + `SampleRate` (48000 / 24000) |
| Duraklama sessizliği modelin hızında | `dotnet\EmaLightning.Core\Playhead.cs` | `Pause * Engine.Rate` | `Request.Rate` (engine.SampleRate) |
| 24 → 48 kHz akış halinde yükseltme | `dotnet\EmaLightning.Core\Audio.cs` | yok | `Upsampler2` |
| 24 kHz modelde yükseltme sonra mevcut 48 kHz zinciri | `dotnet\EmaLightning.Core\Ema.cs` | Receive(request, sampleRate, voice) | Receive(request, sourceRate, sampleRate, voice) |
| Konuşmacı (model seti) seçimi | `dotnet\EmaLightning.Server\Program.cs`, `appsettings.json` | tek `Ema:ModelDirectory` | `Ema:Speakers {female, male}`, `DefaultSpeaker`, `GET /voices`, istekte `speaker` |
| Speaker seçimi; eski "Voice" → "Pitch / formant preset" | `dotnet\EmaLightning.Server\wwwroot\index.html` | yok | eklendi |
| T3/T4: hizalama skor filtresi (`--min-score`), `--workers`, `--ctc/--decoder24` | `training\common\acoustic.py`, `t3_teacher.py`, `t4_distill.py` | — | eklendi |
| T2 Viterbi int8 taşma düzeltmesi | `training\t2_align.py` | `state -= back[t, state]` | `state = int(state) - int(back[t, state])` |
| T5 değerlendirme (yeni) | `training\t5_evaluate.py` | yok | yeni |
| Erkek model seti (üretilen) | `models\male\` (text/sound/decoder.onnx, ema_meta.json) | yok | yeni |

## 2026-10-07 — Erkek ses eğitimi önerisi

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| Eğitim fazları (T0–T5), kararlar, riskler, açık sorular; kod yok | `TRAINING_PROPOSAL.md` | yok | yeni |

## 2026-10-07 — Ses dönüştürücü A yolu: WORLD (native DLL + P/Invoke), kullanıcı isteği

Kaynak: `github\World` (git clone mmorise/World, modified BSD, değiştirilmedi).
Yedekler: `20261007_032345_EmaLightning.Core.csproj.bck`, `20261007_032345_Playhead.cs.bck`, `20261007_032345_Ema.cs.bck`, `20261007_032345_Program.cs.bck`, `wwwroot\20261007_032345_index.html.bck`

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| WORLD sarmalayıcı: Harvest/DIO+StoneMask → CheapTrick → D4C → F0×pitch, zarf+aperiodiklik×formant → Synthesis | `native\world_shift\world_shift.cpp` | yok | yeni |
| VS 2022 x64 ile `world_shift.dll` derlemesi | `native\world_shift\build.bat` | yok | yeni |
| DLL ve WORLD lisansını çıktıya kopyala | `dotnet\EmaLightning.Core\EmaLightning.Core.csproj` | yok | `None Include ... CopyToOutputDirectory` |
| P/Invoke + `VoiceMethod` (Harvest, Dio, Psola) | `dotnet\EmaLightning.Core\WorldShifter.cs` | yok | yeni |
| Cümle sonu işareti (pencereler + duraklama sonrası) | `dotnet\EmaLightning.Core\Playhead.cs` | Send: pencere, duraklama | + `PieceEnd` |
| `method` parametresi (varsayılan Harvest); WORLD cümle bazında PieceEnd'de işler, PSOLA pencere bazında | `dotnet\EmaLightning.Core\Ema.cs` | yalnız PSOLA | Harvest/Dio/Psola |
| JSON `method`: harvest/dio/psola | `dotnet\EmaLightning.Server\Program.cs` | yok | `ParseMethod` |
| Method seçimi (WORLD · Harvest / WORLD · DIO / PSOLA) | `dotnet\EmaLightning.Server\wwwroot\index.html` | yok | eklendi |

## 2026-10-07 — Faz 2: normalizer-tr C# tam portu

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| normalizer-tr 0.4.0 (Rust) tam port, .NET 8 kütüphane, namespace `NormalizerTr` | `dotnet\EmaLightning.Normalizer\*` (28 .cs + csproj) | yok | yeni proje |
| Port'u Rust deposundaki beklenen çıktılarla karşılaştıran konsol aracı (test projesi değil) | `dotnet\EmaLightning.Normalizer.Parity\*` | yok | yeni proje |

## 2026-10-07 — Faz 3: Core (ONNX Runtime CPU)

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| Kütüphane projesi, `Microsoft.ML.OnnxRuntime` 1.26.0 (onaylı paket) | `dotnet\EmaLightning.Core\EmaLightning.Core.csproj` | yok | yeni |
| frontend.py portu (NormalizerTr fallback + alfabe) | `dotnet\EmaLightning.Core\Frontend.cs` | yok | yeni |
| chunker.py portu | `dotnet\EmaLightning.Core\Chunker.cs` | yok | yeni |
| engine.py portu (plan/think/decode ONNX, .NET RNG) | `dotnet\EmaLightning.Core\Engine.cs` | yok | yeni |
| scheduler.py (Playhead) portu | `dotnet\EmaLightning.Core\Playhead.cs` | yok | yeni |
| audio.py portu (Resampler, WAV) | `dotnet\EmaLightning.Core\Audio.cs` | yok | yeni |
| api.py portu (SayAsync/SayManyAsync/StreamAsync, batch ölçüm + cache) | `dotnet\EmaLightning.Core\Ema.cs` | yok | yeni |

## 2026-10-07 — Faz 4: Server

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| ASP.NET Core servis: POST /tts (wav), WS /speak (float32 parçalar), CORS açık, açılışta batch ölçümü | `dotnet\EmaLightning.Server\Program.cs`, `.csproj` | yok | yeni |
| Model klasörü ve Kestrel adresi | `dotnet\EmaLightning.Server\appsettings.json` | yok | yeni |

## 2026-10-07 — Test sayfası (kullanıcı isteği)

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| Tarayıcıdan metin gönderip sesi çalmak (Speak = POST /tts, Stream = WS /speak) | `dotnet\EmaLightning.Server\wwwroot\index.html` | yok | yeni |
| `GET /` → index.html (yedek: `20261007_030451_Program.cs.bck`) | `dotnet\EmaLightning.Server\Program.cs` | `app.UseCors();` `app.UseWebSockets();` | araya `app.UseDefaultFiles();` `app.UseStaticFiles();` |

## 2026-10-07 — Beyza modeli: veri hazırlığı + ifade klipleri (kullanıcı onaylı plan)

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| Parquet → wav + metadata çıkarma | `training\extract_parquet.py` | yok | yeni |
| Sentetik Beyza alt kümesi (~100 sa, tohumlu, 48→24 kHz) | `training\s0_synth_subset.py` | yok | yeni |
| s0 tek süreçte yavaştı (~3 dosya/sn) → ProcessPool (varsayılan 8, çalıştırılan 12 işçi); var olan çıktı atlanır | `training\s0_synth_subset.py` | tek döngü, sıralı dönüştürme | `convert()` işçi fonksiyonu + `--workers` |
| Gerçek + sentetik manifest birleştirme (r_/s_ önek, kaynak bazlı val/test) | `training\m0_merge.py` | yok | yeni |
| Uzun kayıtları ≤250 kare/250 harf parçalara bölme (cümle→yan cümle→kelime) | `training\t2_split.py` | yok | yeni |
| Önceki ağırlıktan başlatma | `training\t1_encoder.py`, `training\t2_ctc.py` | — | `--init` parametresi |
| İfade klipleri (emoji + `<etiket>` / `<etiket_NN>` → konuşmacının gerçek kayıtları; eşlenmeyen sessizce atılır) | `dotnet\EmaLightning.Core\Expressions.cs` | yok | yeni |
| Klip parçası desteği (yedek: `20261007_174814_*.bck`) | `dotnet\EmaLightning.Core\Engine.cs` | `Piece` yalnız model parçası | `Piece.Clip` alanı |
| Klip parçaları plan/think/decode dışında, sırasıyla çalınır | `dotnet\EmaLightning.Core\Playhead.cs` | tüm parçalar decode | `Clip != null` olanlar klonlanarak eklenir |
| `<model>\expressions\` yüklenir, metin `Split` ile parçalanır | `dotnet\EmaLightning.Core\Ema.cs` | — | `Expressions.Load`, `ExpressionNames` |
| Teacher'ı yalnız gerçek veriyle ince ayar (kullanıcı seçimi: yol 2; yedek `training\20261007_195934_t3_teacher.py.bck`) | `training\t3_teacher.py` | `--init`/`--source` yok; val batch sabit `range(0, 8*37, 37)` | `--init` (teacher.pt 'ema' ile başlat, yalnız last.pt yoksa), `--source` (manifest 'source' alanına göre train+val seçimi); val < 296 ise adım = len//8 (eski davranış ≥296'da aynı) |
| Beyza son değerlendirmesi kendi CTC'si ve yalnız gerçek test kayıtlarıyla (kullanıcı onayı; yedek `training\20261007_221710_t5_evaluate.py.bck`) | `training\t5_evaluate.py` | `Judge(device, args.decoder24)` (CTC hep erkek); tüm test kayıtları | `--ctc` → `Judge(device, args.decoder24, args.ctc)`; `--source` → test kayıtları manifest 'source' alanına göre |

## 2026-10-07 — Beyza modeli: dışa aktarım ve servis

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| DMD2 6.000. adımda durduruldu (kullanıcı: "1 olsun"); en iyi nokta snap_6000 (40 gerçek cümle CER %3.80) → t4 final biçimiyle | `checkpoints\beyza\student\ema.pt` | yok | snap_6000 'ema' + teacher cfg (dmd2, 4 adım, cfg_baked 4.0) |
| ONNX dışa aktarım (24 kHz, hop 960) | `models\beyza\{text,sound,decoder}.onnx`, `ema_meta.json` | yok | yeni |
| İfade klipleri (BEYZA_EMOTION, yalnız .wav; mp4 alınmadı) | `models\beyza\expressions\*.wav` (60) | yok | kopya |
| Servise Beyza konuşmacısı (yedek: `dotnet\EmaLightning.Server\*_appsettings.json.bck`) | `dotnet\EmaLightning.Server\appsettings.json` | female, male | + `"beyza": "..\..\models\beyza"` |

## 2026-10-07 — BeyzaR: yalnız gerçek veriyle ikinci Beyza konuşmacısı (kullanıcı onayı)

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| Teacher + DMD2 yalnız gerçek Beyza verisiyle (1.096 train / 19 val / 21 test parça) | `data\beyzaR_train\` (manifest, durations: `beyza_train`'den source=r; latents.pt kopya) | yok | yeni |
| Küçük val setinde (19) indeks hatası (yedek `training\20261007_223058_t4_distill.py.bck`) | `training\t4_distill.py` | `range(0, 8 * 37, 37)` | val < 296 ise adım = len//8 (≥296'da eski davranış) |
| BeyzaR öğrenci 1.500. adım (kullanıcı: "1500 gayet güzel"), DMD2 arkada sürüyor | `checkpoints\beyzaR\student\ema_1500.pt` → `models\beyzaR\` (+ `expressions\` Beyza'nınkiyle aynı 60 wav) | yok | yeni |
| Servise BeyzaR konuşmacısı (yedek `dotnet\EmaLightning.Server\20261007_230240_appsettings.json.bck`); servis yeniden başlatıldı | `dotnet\EmaLightning.Server\appsettings.json` | female, male, beyza | + `"beyzaR": "..\..\models\beyzaR"` |

## 2026-10-08 — Depo için doküman temizliği (kullanıcı isteği)

| Sebep | Dosya | Önce | Sonra |
|---|---|---|---|
| Hak/izin notları yayından çıkarıldı (yedekler `20261008_000534_*.md.bck`) | `BEYZA_STATUS.md`, `TRAINING_PROPOSAL.md`, `ASR_PROPOSAL.md`, `MULTI_SPEAKER_PROPOSAL.md` | veri satırlarında hak/izin açıklamaları | kaldırıldı |
| Dokümanlarda yerel yollar gizlendi (yedekler `*_ASR_PROPOSAL.md.bck`, `*_BEYZA_STATUS.md.bck`, `*_TRAINING_PROPOSAL.md.bck`) | `ASR_PROPOSAL.md`, `BEYZA_STATUS.md`, `TRAINING_PROPOSAL.md` | sürücü harfli veri/kayıt/proje yolları | `<datasets>\…`, `<recordings>\…`, `<repo>` |
| Veri seti adı kaldırıldı (yedekler 20261008_003600_*.bck) | `BEYZA_STATUS.md`, `TRAINING_PROPOSAL.md` | ad geçen satır/ifade | kaldırıldı |
