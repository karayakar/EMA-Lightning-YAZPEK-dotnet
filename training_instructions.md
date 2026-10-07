# Eğitim talimatları — yeni konuşmacı

Bu belge, yeni bir konuşmacıyı sıfırdan eğitip servise eklemek için çalıştırılacak betikleri, parametreleri ve sırayı anlatır. Tüm komutlar `training\` klasöründen çalıştırılır. `<ad>` yerine konuşmacı adını yazın (ör. `zehra`).

## 0. Ortam kurulumu

İki ayrı Python ortamı kullanılır. Bu projede kullanılan sürümler: Python 3.12.0, torch 2.14.1 (eğitimde CUDA 13.0 derlemesi `cu130`, dışa aktarımda CPU), numpy 2.5.3, ema-lightning 1.0.1, normalizer-tr 0.4.0.

| Ortam | Klasör | Paketler |
|---|---|---|
| Eğitim (GPU) | `training\.venv` | torch (CUDA), ema-lightning, normalizer-tr, tensorboard, pyarrow |
| ONNX dışa aktarım (CPU) | `onnx-export\.venv` | torch (CPU), ema-lightning, normalizer-tr, onnx, onnxruntime |

### Eğitim ortamı (`training\.venv`)
```
cd training
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install ema-lightning normalizer-tr tensorboard pyarrow
```
CUDA derlemesini ekran kartı sürücünüze göre seçin (`cu130` için sürücünün CUDA 13.x desteklemesi gerekir; `nvidia-smi` sağ üstte gösterir). Kontrol:
```
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Dışa aktarım ortamı (`onnx-export\.venv`)
```
cd onnx-export
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install ema-lightning normalizer-tr onnx onnxruntime
```

### Orijinal ağırlıklar
Hugging Face `canberkkkkkk/ema-lightning` deposu (`ema.pt`, `decoder.pt`, `config.json`) proje kökündeki `ema-lightning\` klasörüne indirilir:
```
.venv\Scripts\python.exe -m pip install huggingface_hub
.venv\Scripts\hf.exe download canberkkkkkk/ema-lightning --local-dir ..\ema-lightning
```
Encoder, decoder ve teacher bu ağırlıklardan başlar; erkek modelin CTC/encoder'ından ince ayar yapılacaksa `checkpoints\male\` de gerekir (depoda yok).

Windows'ta Türkçe çıktılar için `set PYTHONIOENCODING=utf-8` önerilir.

## 1. Veri biçimi

Veri klasörü: `wavs\*.wav` + `metadata.csv`. Her satır: `dosya.wav| metin`. Ses mono; 24 kHz değilse T0 dönüştürür. Parquet (Hugging Face) veri seti önce `extract_parquet.py --dataset <klasör>` ile wav + metadata'ya çıkarılır.

## 2. Akış (özet)

| # | Adım | Betik | Çıktı |
|---|---|---|---|
| T0 | Veri hazırlığı | `t0_prepare.py` | `data\<ad>\manifest.jsonl`, `t0_report.md` |
| M0 | (İsteğe bağlı) birden çok kaynağı birleştir | `m0_merge.py` | `data\<ad>_all\manifest.jsonl` |
| T2a | CTC hizalayıcı | `t2_ctc.py` | `checkpoints\<ad>\ctc\ctc.pt` |
| T2b | Hizalama (kelime süreleri) | `t2_align.py` | `durations.jsonl`, `t2_report.json` |
| T2c | Uzun kayıtları böl (≤10 sn) | `t2_split.py` | `manifest_split.jsonl`, `durations_split.jsonl` |
| T1a | Encoder | `t1_encoder.py` | `checkpoints\<ad>\encoder\encoder.pt` |
| T1b | 24 kHz decoder | `t1_decoder.py` | `checkpoints\<ad>\decoder\decoder24.pt` |
| T3a | Latentler | `t3_latents.py` | `latents.pt` |
| T3 | Teacher (flow matching) | `t3_teacher.py` | `checkpoints\<ad>\teacher\teacher.pt` |
| T4 | DMD2 damıtma (4 adım) | `t4_distill.py` | `checkpoints\<ad>\student\ema.pt` |
| T5 | Değerlendirme | `t5_evaluate.py` | `t5_report.json`, `samples\test\` |
| E | ONNX dışa aktarım | `onnx-export\export.py` | `models\<ad>\` |
| S | Servise ekle | `appsettings.json` | `speaker: <ad>` |

T1a / T2a aynı anda, T1b / T3 / T4 de birlikte çalışabilir (24 GB GPU'da sığar). Her betik `last.pt` varsa kaldığı yerden devam eder.

## 3. Adımlar

### T0 — veri hazırlığı
```
.venv\Scripts\python.exe t0_prepare.py --data <veri klasörü> --out ../data/<ad>
```
Kırpma, −23 dBFS seviye, bant/hız ölçümü, tohumlu %1 val + %1 test. `t0_report.md`'yi okuyun (reddedilenler, bant, konuşma hızı).

### M0 — birleştirme (yalnız birden çok kaynak varsa)
```
.venv\Scripts\python.exe m0_merge.py --part r,../data/<ad>,10 --part s,../data/<ad>_synth,200 --out ../data/<ad>_all
```
`önek,klasör,val_test_adedi`. Kayıtlara `source` alanı (`r`, `s`) eklenir; sonraki adımlarda `--source` ile seçilebilir.

### T2a — CTC hizalayıcı
```
.venv\Scripts\python.exe t2_ctc.py --manifest ../data/<ad>/manifest.jsonl --out ../checkpoints/<ad>/ctc --init ../checkpoints/male/ctc/ctc.pt --epochs 4 --lr 3e-4
```
Mevcut bir CTC'den (`--init`) ince ayar hızlıdır. Sıfırdan: `--init` olmadan, `--epochs 6 --lr 1e-3`.

### T2b — hizalama
```
.venv\Scripts\python.exe t2_align.py --manifest ../data/<ad>/manifest.jsonl --ctc ../checkpoints/<ad>/ctc/ctc.pt --out ../data/<ad>
```
`t2_report.json` → `score_percentiles`. **`"1"` (alt %1) değeri T3/T4'teki `--min-score` olur.**

### T2c — uzun kayıtları bölme
```
.venv\Scripts\python.exe t2_split.py --data ../data/<ad>
```
T3/T4 sabit dosya adları bekler. Ayrı bir eğitim klasörü hazırlayın:
```
mkdir ..\data\<ad>_train
copy ..\data\<ad>\manifest_split.jsonl ..\data\<ad>_train\manifest.jsonl
copy ..\data\<ad>\durations_split.jsonl ..\data\<ad>_train\durations.jsonl
```

### T1a — encoder
```
.venv\Scripts\python.exe t1_encoder.py --manifest ../data/<ad>/manifest.jsonl --out ../checkpoints/<ad>/encoder --init ../checkpoints/male/encoder/encoder.pt --steps 10000
```
Orijinal 48 kHz decoder dondurulmuş kalır. `val mel` platoya girince (genelde 5k–10k) yeterlidir.

### T1b — 24 kHz decoder
```
.venv\Scripts\python.exe t1_decoder.py --manifest ../data/<ad>/manifest.jsonl --encoder ../checkpoints/<ad>/encoder/encoder.pt --out ../checkpoints/<ad>/decoder --steps 15000 --seconds 0.96
```
Sentetik veri varsa decoder'ı **yalnız gerçek** kayıtlarla eğitin (gerçek kayıtlardan bir manifest süzün). Doğrulama için manifestte en az ~6 val kaydı olmalı.

### T3a — latentler
```
.venv\Scripts\python.exe t3_latents.py --manifest ../data/<ad>_train/manifest.jsonl --encoder ../checkpoints/<ad>/encoder/encoder.pt --out ../data/<ad>_train/latents.pt
```

### T3 — teacher
```
.venv\Scripts\python.exe t3_teacher.py --data ../data/<ad>_train --out ../checkpoints/<ad>/teacher --steps 10000 --min-score=<T2b alt %1> --ctc ../checkpoints/<ad>/ctc/ctc.pt --decoder24 ../checkpoints/<ad>/decoder/decoder24.pt
```
- Orijinal `ema.pt`'den başlar. Başka bir teacher'dan devam: `--init <teacher.pt>`.
- Yalnız bir kaynak: `--source r`.
- **Az veri (~2 sa)**: `--steps 3000 --every 500`; 1.5k–2.5k civarı en iyi olabilir. Fazlası ezberler.
- Negatif değer `=` ile verilir: `--min-score=-0.952`.

### T4 — DMD2 damıtma
```
.venv\Scripts\python.exe t4_distill.py --data ../data/<ad>_train --teacher ../checkpoints/<ad>/teacher/teacher.pt --out ../checkpoints/<ad>/student --steps 8000 --every 500 --min-score=<T2b alt %1> --ctc ../checkpoints/<ad>/ctc/ctc.pt --decoder24 ../checkpoints/<ad>/decoder/decoder24.pt
```
- Bitince `student\ema.pt` (dışa aktarılacak dosya) yazılır.
- Erken durdurulursa `ema.pt` oluşmaz; ara noktadan dışa aktarma için `last.pt`'nin `ema` ağırlığı teacher'ın `cfg` / `vocab` bilgisiyle `ema.pt` biçimine kaydedilmelidir (bkz. `t4_distill.py` → `save(final=True)`).
- `last.pt` her değerlendirmede üzerine yazılır; ara noktaları saklamak için kopyalayın.

### T5 — değerlendirme
```
.venv\Scripts\python.exe t5_evaluate.py --data ../data/<ad>_train --teacher ../checkpoints/<ad>/teacher/teacher.pt --student ../checkpoints/<ad>/student/ema.pt --decoder24 ../checkpoints/<ad>/decoder/decoder24.pt --ctc ../checkpoints/<ad>/ctc/ctc.pt --out ../checkpoints/<ad>
```
Yalnız gerçek test kayıtları: `--source r`. Noktalamasız CER: kayıtlar / teacher 16 adım / öğrenci 4 adım + GPU RTF.

### E — ONNX dışa aktarım (`onnx-export` klasöründen)
```
.venv\Scripts\python.exe export.py --acoustic ../checkpoints/<ad>/student/ema.pt --decoder ../checkpoints/<ad>/decoder/decoder24.pt --out ../models/<ad>
```
İfade klipleri (isteğe bağlı): `models\<ad>\expressions\<kategori>_NN.wav` (mono PCM16, modelin hızında, 24 kHz).

### S — servis
`dotnet\EmaLightning.Server\appsettings.json` → `Ema:Speakers` içine `"<ad>": "..\\..\\models\\<ad>"` ekleyin ve servisi yeniden başlatın. `GET /voices` listede göstermelidir.

## 4. Dikkat

| Konu | Not |
|---|---|
| `--decoder24` / `--ctc` | T3/T4/T5'te **her zaman açıkça verin**. Verilmezse erkek decoder/CTC kullanılır ve CER anlamsız çıkar (ör. %40 yerine gerçek %6). |
| 8 cümlelik ara CER | Gürültülüdür; karar için 40+ cümlelik ölçüm ve dinleme kullanın. |
| CTC ölçümü | CTC sentetik ağırlıklı eğitildiyse gerçek sese daha yüksek CER verir; anlaşılırlık için iyi, doğallık için kulak testi şart. |
| Sentetik + gerçek | Sentetik veri telaffuzu güçlendirir ama sesi "sentetik"leştirir; ses kimliği için gerçek veri belirleyicidir. |
| Az gerçek veri | Gerçek-only teacher/DMD2 ezberlemeye hızlı gider; sık ara nokta + dinleme ile en iyi adımı seçin. |
| GPU belleği | T1b + T3 (+ T4) eşzamanlı 24 GB'a sığar; başka GPU uygulamaları kapalı olmalı. |
| Windows | `--workers` ≤ 61; T3/T4'te `--workers 0` (latent sözlüğü her işçiye kopyalanır). |

## 5. Sentetik veri üretimi (isteğe bağlı)

| Betik | Amaç |
|---|---|
| `s0_synth_subset.py --source <klasör> --metadata <csv> --out ../data/<ad>_synth_src --count N --workers 12` | Büyük sentetik setten tohumlu alt küme, 48→24 kHz |
| `s1_omnivoice.py --texts <metin> --out <klasör> --instruction "female, whisper"` | OmniVoice HTTP (GPU, `127.0.0.1:8080`) ile talimatlı veri seti; aynı ayar + tohum aynı sesi verir |
