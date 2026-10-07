# Eğitim durumu (gece çalışması)

## ✅ SONUÇ (2026-10-07 sabah)

Erkek ses modeli eğitildi, ONNX'e aktarıldı, servise eklendi. **Mimari ve boyut orijinalle aynı (5.56M + decoder, 4 adım).**

| Ölçüt (200 test cümlesi, noktalamasız CER, kendi CTC'miz) | Değer |
|---|---|
| Gerçek kayıtlar | %1.65 |
| Teacher (16 adım) | %1.00 |
| **Son model — öğrenci (4 adım)** | **%0.82** |
| F0 | ~112 Hz (kadın modeli ~245 Hz) |
| CPU hızı (servis, uçtan uca) | RTF ~0.17 (kadın ~0.16) → gerçek zamandan ~6× hızlı; erkek aynı metni daha yavaş okur (veri temposu) |
| Stream (arayüz) | 13.5 sn ses 2.3 sn'de, ilk parça 1.4 sn |

**Kullanım:** `http://localhost:5180/` → "Speaker (model)" = Male. API: `"speaker": "male"`. `GET /voices`.
**Dosyalar:** `models\male\` (ONNX), `checkpoints\male\student\ema.pt`, `checkpoints\male\decoder\decoder24.pt`.
**Dinle:** `test-output\trained-male-*.wav`, `checkpoints\male\samples\test\student4\` ve `teacher16\` (12'şer test cümlesi).

**Bilinen sınırlar:** 24 kHz kaynak → 12 kHz bant; okuma tarzı veri gibi (sözlük tanımı ritmi); değerlendirme kendi CTC'miz ile (Whisper değil); dinleme testi yapılmadı (kullanıcıda).

**Geçici dosyalar (silinebilir, ben silmedim):** `data\male_smoke\`, `checkpoints\smoke\`, `checkpoints\male\ctc_smoke.pt`, `checkpoints\male\decoder\decoder24_interim.pt`.

Kullanıcı 2026-10-07 gece: "ben yatıyorum, sen devam et, soru sorma, kontrol sende."
Kurallar (kendi koyduğum): kaynak veri (`100HRSKARAY`) yalnız okunur; her karar burada; dinleme kapıları yerine
ölçülebilir kriter (loss, CTC CER) + örnek klasörleri; kötü sonuçla bir sonraki faza geçilmez.

## Zaman çizelgesi

| Saat | Olay |
|---|---|
| — | T0 başladı (manifest + rapor) |
| T1a durduruldu | 5.000. adımda (val mel 2.5k: 0.846 → 5k: 0.849, plato). `checkpoints\male\encoder\encoder.pt` |
| T2 CTC başladı | 6 epoch; epoch 1 val CER **%7.76** |
| T1b başladı | CTC ile eşzamanlı |
| T1b 2.5k | val mel 0.788; LSD ort. 11.5 dB (T1a'da 13–15); F0 6 örneğin 5'inde doğru (~100 Hz) |
| CTC bitti | val CER epoch 1→6: %7.76, 5.81, 5.23, 4.88, 4.28, **3.59**. `checkpoints\male\ctc\ctc.pt` |
| T2b hizalama + T3 latentleri başladı | sırayla (ilk koşu int8 taşma hatası → düzeltildi, yeniden) |
| Latentler | 95.959 kayıt, 8.59 M kare, `data\male\latents.pt` (1.1 GB) |
| T1b 5k | val mel 0.703; F0 6/6 doğru; LSD ort. 12.1 dB |
| T3 başladı | T1b ile eşzamanlı; ara örnekler T1b 5k ara decoder'ıyla (`decoder24_interim.pt`) |
| T3 2.5k | **CER %1.44** (8 val cümlesi, metinden üretim, 16 adım CFG 4); F0 6/8 örnekte 101–114 Hz (2'si muhtemel ölçüm oktav hatası); hız 12–16 harf/sn (veri medyanı 14.2) |
| T3 durduruldu | 10k adımda (CER 2.5k/5k/7.5k: %1.44, 10k: **%1.15**; fm platosu ~0.36). Kalan hatalar noktalama/kelime arası (ör. "ağzın"→"ağzım", "biri lep"→"birilep"). `checkpoints\male\teacher\teacher.pt` |
| T1b 7.5k | val mel 0.690; LSD 10.9 dB (iyileşiyor, 15k'ya kadar sürüyor) |
| T4 başladı | DMD2, ara decoder T1b 7.5k; ~1.1 adım/sn (T1b ile eşzamanlı) |
| T4 0 / 500 | 4 adım CER: damıtma öncesi %3.46 → 500. adım **%1.73** (teacher 16 adım: %1.15) |
| T4 1k / 1.5k | CER %2.02 / %2.31 — incelendi: fark noktalama/boşluk (virgül, "..."→".", "beş aret"); gerçek telaffuz hatası yok (damıtma öncesi "çedeci çet eden" vardı, düzeldi). RMS 0.06 → 0.08 (teacher seviyesi). Son değerlendirme noktalamasız CER + daha geniş test seti ile yapılacak |
| T1b 10k | val mel 0.612; LSD 10.9 dB |
| T4 2k–4k | CER %2.02, 1.73, 1.44, 1.73, **1.15** (4k = teacher 16 adım seviyesi) |
| T1b 12.5k | val mel 0.610 (plato) |
| Karar | T4 8k yerine T1b bitince (~5k) durdurulur; T5 öğrenci ≈ teacher değilse kaldığı yerden devam |
| T1b bitti | 15k; val mel 0.622. `decoder24.pt` |
| T4 durduruldu | 6k (CER 4.5k–6k: %2.02, 1.44, 1.44, 1.44). Öğrenci EMA → `student\ema.pt` |
| T5 | 200 test: kayıt %1.65, teacher16 %1.00, **öğrenci4 %0.82** (noktalamasız CER). GPU RTF teacher 0.026 / öğrenci 0.008 |
| Export | `models\male\` (sample_rate 24000, hop 960); ONNX ↔ PyTorch: latent max 7.7e-5, ses 1.4e-6 |
| Servis | female + male yüklendi; male 48/24/16 kHz HTTP 200; CPU RTF ~0.17; arayüz stream çalıştı |
| T2b bitti | 95.959/95.959 hizalandı, 0 hata, 250 kareyi aşan kelime yok; skor medyan −0.076, p1 −0.332. `data\male\durations.jsonl` |
| T0 bitti | 95.959 kayıt / **95.92 saat** kullanılabilir; train 94.095, val 932, test 932; long (>10 sn veya çok parçalı) 2.744; ret 355 (173 metadata satırı boş metin `571\|`, 177 wav metinsiz, 5 metni frontend'den sonra boş). Bant: ≥10 kHz 89.869, 8–10 kHz 5.820, <8 kHz 270. Konuşma hızı medyan 14.2 harf/sn. Rapor: `data\male\t0_report.md` |

## Kararlar (kullanıcıya sorulmadan alınanlar)

| Konu | Karar | Neden |
|---|---|---|
| Hız aykırı 949 kayıt + dar bant 270 kayıt | T1'de (yalnız ses) tutuldu; T2/T3 öncesi CTC güven skoruyla yeniden değerlendirilecek | Ses kalitesi sorunu değil, çoğu tek kelime + uzun sessizlik ya da normalizer açılımı |
| Long 2.744 kayıt | T1'de rastgele kesitlerle kullanılır; T3 için T2 hizalamasından sonra karar | Transkript gerekmiyor |
| GPU belleği | LTX Desktop (python, PID 4856) ~20 GB VRAM tutuyor; **kapatılmadı / boşaltılmadı** (kullanıcının uygulaması). Eğitimler kalan ~4 GB'a sığdırıldı; fazlar sırayla, eşzamanlı değil | İlk T1a koşusu VRAM taşıp sistem belleğine döküldü: 1.3 adım/sn |
| T1a batch | 16 × 1.92 sn (2.3 GB, ~97 ms/adım) | 16 × 2.88 sn VRAM'e sığmadı |
| T3/T4 veri filtresi | Hizalama skoru < −0.332 (alt %1, ~960 kayıt) eğitimden çıkarıldı (silinmedi); `--min-score` | En kötüler normalizer/ses uyuşmazlığı: "unikod u artı…", harf harf heceleme, İngilizce kelimeler |
| LTX Desktop | Kullanıcı kapattı (GPU 3 GB kullanımda, 21 GB boş). Ben dokunmadım | — |
| T1b | Tam HiFi-GAN MPD (1024 kanal) geri; batch 16 × 0.96 sn, bf16, **15k adım** (ince ayar), CTC ile eşzamanlı | Bellek artık yeterli (7.2 GB); ~428 ms/adım |
| T1a uzunluğu | 30k yerine ≤10k adım (plato görülürse daha erken) | 2.5k'da: val mel 0.846, LSD 13–15 dB, F0 kaymaları; sınır dondurulmuş kadın decoder'ı — kaliteyi T1b düzeltir. Kadın referans yeniden sentezi ~sessiz (encoder yalnız erkekle eğitildi, beklenen) |

## Dinleme örnekleri

| Faz | Klasör |
|---|---|
| T1b decoder (gerçek ses → yeniden sentez) | `checkpoints\male\decoder\samples\<adım>\val*_orig.wav` / `val*_recon.wav` |
| T3 teacher (metinden, 16 adım) | `checkpoints\male\teacher\samples\<adım>\val*.wav` + `texts.txt` |
| T4 öğrenci (metinden, 4 adım) | `checkpoints\male\student\samples\<adım>\val*.wav` + `texts.txt` |
