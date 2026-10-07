# EMA Lightning — Erkek Ses Eğitimi Önerisi (TRAINING_PROPOSAL)

Durum: **ÖNERİ — onay bekliyor, kod yok.** Tarih: 2026-10-07

## 0. Hedef ve kesin şartlar

| Şart | Karşılığı |
|---|---|
| Gerçek erkek sesi (F0/formant kaydırma değil) | 100 saatlik tek erkek konuşmacı verisiyle model eğitimi |
| **Hız kaybı yok** | Son model bugünküyle **aynı mimari, 8.6M parametre, 4 adım** (DMD2 ile yeniden damıtma zorunlu) |
| Ticari kullanım serbest | Yalnız kullanıcının verisi (ticari serbest) + kendi eğittiğimiz modeller; dış model/veri yok |
| Çıkarım tarafı değişmesin | Aynı `export.py` → `models\male\`; C# pipeline aynı, servise yalnız `voice` seçimi eklenir |

Donanım: RTX 3090 (24 GB), bu makine (Windows). Veri: ~100 saat, tek erkek, wav 24 kHz, **24 kHz'te eğitilir** (kullanıcı kararı, 2026-10-07; önceki 48 kHz kararının yerine). Erkek decoder'ı 24 kHz çıkışlıdır (hop 960); bant genişliği kaynakla aynı (≤12 kHz). Kaynak gerçek 24 kHz: 200 örnekte −60 dB sınırı medyan 11.96 kHz, p10 10.4 kHz, en düşük 7.6 kHz.

## 1. Elimizde olan / olmayan (checkpoint incelemesi)

| Var | Yok (bu öneride tasarlanacak) |
|---|---|
| `ema.pt`: damıtılmış öğrenci (`model`, `ema`), cfg: d=224, 6 DiT katmanı, 8 head, `text_conv` 4, **`text_attn` 0**, `align_heads` 4, `pos_scale` 4.0, lookback/lookahead 1, `dur_hidden` 256, `letter_pos` true, `shared_ada` true, DMD2 4 adım `times` [0, .25, .5, .75], **`cfg_baked` 4.0** | Teacher (`step_0590000.pt`), fake-score ağı, optimizer durumu |
| `decoder.pt`: HiFi-GAN üretici `G` (weight-norm g/v), rates (8,6,5,2,2,2), hop 1920, 48 kHz | **Latent encoder** (ses → 64-b @25 Hz), discriminator'lar |
| `model.py`, `decoder.py`, `engine.py` (çıkarım) | Eğitim kodu, loss'lar, hiperparametreler, süre hedeflerinin nasıl üretildiği |

Not: mimari şemada metin kodlayıcı "4×ConvNeXt + 2×MHSA" yazıyor; dağıtılan checkpoint'te `text_attn: 0`. **Eğitimde checkpoint config'i esas alınır.**

Aşağıdaki loss'lar, ağ yapıları (encoder, CTC hizalayıcı, discriminator) ve hiperparametreler **benim tasarımımdır**; orijinal eğitimle aynı olduğu iddia edilmez.

## 2. Klasör düzeni

```
<repo>\
  training\                 eğitim kodu (Python)
    .venv\                  ayrı ortam (CUDA torch) — onnx-export\.venv'e dokunulmaz
    common\                 mel/STFT, veri okuyucu, frontend köprüsü, loss'lar, discriminator'lar
    t0_prepare.py  t1_encoder.py  t1_decoder.py  t2_ctc.py  t2_align.py
    t3_teacher.py  t4_distill.py  t5_evaluate.py
  data\male\
                            (ses kopyalanmaz: kırpma noktaları + kazanç manifest'te, orijinal 24 kHz okunur)
    manifest.jsonl          {id, wav, text_raw, text_model, seconds, split}
    latents\  durations\    T1/T2 çıktıları
  checkpoints\male\         encoder / decoder / ctc / teacher / student
  models\male\              T5: text.onnx, sound.onnx, decoder.onnx, ema_meta.json
```

## 3. Fazlar

Her fazın sonunda **kullanıcı onayı / dinleme kapısı** var; onaysız bir sonrakine geçilmez.

### T0 — Veri hazırlığı
- Girdi: `<datasets>\100HRSKARAY\wavs` + `metadata.csv` (§7).
- 24 kHz korunur; format denetimi (PCM16 mono 24 kHz), baş-son sessizlik kırpma (−45 dB, 100 ms pay), aktif konuşma RMS'i −23 dBFS + tepe ≤0.95 — kırpma noktaları ve kazanç manifest'e yazılır, ses kopyalanmaz. Dosya başına bant sınırı (−60 dB) raporlanır.
- Metin: **çıkarımdakiyle aynı frontend** (normalizer-tr `fallback` + alfabe + chunker kuralları) → `text_model`. Model alfabesine (49 sembol) girmeyen satırlar raporlanır, sessizce düşürülmez.
- 10 sn / 250 harf üstü kayıtlar: noktalama sınırından bölünür (hizalama T2'den sonra yapılabildiği için ilk turda yalnız rapor; bölme T2 sonrası).
- Bölüm: train %98 / val %1 / test %1 (konuşmacı tek; rastgele, tohumlu).
- Çıktı: `manifest.jsonl` + istatistik raporu (toplam süre, uzunluk dağılımı, düşen satırlar).

### T1 — Latent encoder ve erkek decoder'ı
Amaç: erkek kayıtlarından, mevcut akustik modelin latent uzayıyla **uyumlu** 64-b @25 Hz hedefler üretmek ve bu latentlerden erkek sesini 48 kHz üretebilen decoder.

**T1a — Encoder (orijinal 48 kHz decoder dondurulmuş)**
- Ağ: 24 kHz log-mel (100 bant, n_fft 1024, hop 240 → 100 kare/sn) → 1B konvolüsyon ResNet → stride-2 ×2 → 25 Hz → 64 kanal.
- Loss 24 kHz'te hesaplanır: orijinal decoder çıktısı (48 kHz) ×2 alçaltılıp girdiyle karşılaştırılır.
- Loss: `D(E(x))` ile `x` arasında çok çözünürlüklü STFT + mel L1. Ek olarak latent momentleri eşleme: mevcut akustik modelle (kadın) üretilmiş latentlerin boyut başına ortalama/std'sine yakınlık (uzay uyumu için).
- Kontrol: kadın örnekleri (`ema-lightning\assets`) encode→decode yeniden sentezi orijinale yakın olmalı.

**T1b — Decoder ince ayarı (erkek)**
- **24 kHz decoder**: rates (8, 6, 5, 2, 2) → hop 960. İlk 5 yükseltme katmanı ve resblock'ları `decoder.pt`'den alınır; son ×2 katmanı çıkarılır, `post` konvolüsyonu yeniden başlatılır. HiFi-GAN tarzı GAN: MPD + MRD discriminator (sıfırdan), LSGAN + feature matching + mel L1.
- Encoder küçük öğrenme oranıyla ortak ya da dondurulmuş (ilk koşuda karşılaştırılır).
- **Kapı:** val setinde encode→decode yeniden sentezini dinlersin (erkek tınısı, 12 kHz bant).

### T2 — Hizalama (kelime süreleri)
Modelin zaman çizelgesi **kelime bazlı**dır (`engine.plan`: harf süreleri kelimede toplanıp kare sayısına çevrilir, harf konumunu aligner bulur). Bu yüzden hedef olarak yalnızca **kelime başına kare sayısı (25 Hz)** gerekir. Boşluk karakteri önceki kelimeye ait olduğundan (`engine.piece`), kelimeler arası sessizlik önceki kelimenin karelerine girer — çıkarımdaki ile aynı.
- Kendi CTC hizalayıcımız: mel → küçük konvolüsyon + çift yönlü katmanlar → 49 sembol + blank. **Yalnız kullanıcının 100 saatiyle eğitilir** (dış ASR modeli / lisans riski yok).
- Viterbi zorlamalı hizalama → harf sınırları → kelime sınırları → kelime kare sayıları (kümülatif yuvarlama, toplam = latent kare sayısı).
- Uzun kayıtlar burada noktalama sınırlarından ≤10 sn parçalara bölünür.
- **Kapı:** rastgele örneklerde sınırların görsel/sesli kontrolü + val CER raporu.

### T3 — Teacher (flow matching)
- Mimari: akustik modelin aynısı (cfg yukarıda). **Mevcut `ema` ağırlıklarından başlar** (Türkçe metin/hizalama bilgisini taşır).
- Koşul: gerçek kelime süreleriyle zaman çizelgesi (`fw`, `fp`) → aligner → `c`.
- Flow matching (örnekleyiciyle uyumlu): `z_t = (1−t)·ε + t·z₁`, hedef hız `z₁ − ε`, maskeli MSE. Sınıflandırıcısız rehberlik için koşul düşürme (p≈0.1, `c = 0`).
- Süre: harf süresi tahmini kelimede toplanır, `log(1+·)` uzayında MSE (yalnız kelime toplamı denetlenir).
- Örnekleme (değerlendirme): 16 orta nokta adımı, CFG 4.0 (README'deki teacher ayarı).
- **Kapı:** teacher örneklerini dinlersin + T2 CTC ile CER.

### T4 — DMD2 damıtma (zorunlu, hız şartı)
- Öğrenci: teacher'dan başlar, **4 adım, t ∈ {0, ¼, ½, ¾}**, örnekleyici `sound_stage` ile birebir (adım başı yeni gürültüyle yeniden gürültüleme).
- Gerçek skor: dondurulmuş teacher + CFG 4.0 (öğrenciye gömülür → `cfg_baked` 4.0).
- Sahte skor: teacher kopyası, öğrenci çıktılarında çevrimiçi flow-matching ile güncellenir (iki zaman ölçekli: sahte skor öğrenciden daha sık).
- Dağılım eşleme gradyanı + sahte skor özelliklerinde küçük bir GAN başlığı (DMD2 tarzı); çok adımlı öğrencide geriye simülasyonla ara adım girdileri.
- Metin kodlayıcı / süre / aligner teacher'dan alınır ve dondurulur; yalnız DiT eğitilir.
- **Kapı:** öğrenci (4 adım) teacher'a yakın olmalı; dinleme + CER.

### T5 — Değerlendirme, export, servis
- Test seti: CER (T2 CTC ile), süre/tempo istatistiği, dinleme. (Whisper tabanlı WER istenirse ayrı paket onayı — §7-3.)
- `onnx-export\export.py`'ye ağırlık/çıktı klasörü parametresi → `models\male\` (aynı ONNX girdileri).
- Servis: `Ema` örneği ses başına (female/male); istekte `voice`. Pitch/formant araçları opsiyonel kalır.
- C#: Engine örnekleme hızını `ema_meta.json`'dan okur (erkek 24 kHz). Resampler yalnız tamsayı oranlı: 24 kHz modelden 48000 ve 16000 istekleri için çözüm (yükseltme / kesirli oran) T5'te ayrıca sorulur.
- Hız: aynı mimari → bugünkü CPU ölçümleriyle aynı olması beklenir (T5'te ölçülür).

## 4. Hiperparametre başlangıçları (ilk koşuda ölçülüp düzeltilecek)

| Faz | Optimizer | Başlangıç LR | Batch (yaklaşık) | Hassasiyet |
|---|---|---|---|---|
| T1a encoder | AdamW | 2e-4 | 16 × 4 sn | bf16 |
| T1b decoder GAN | AdamW (β 0.8/0.99) | 1e-4 (G), 1e-4 (D) | 16 × 2 sn | fp32 (GAN kararlılığı) |
| T2 CTC | AdamW | 1e-3 | ~64 kayıt | bf16 |
| T3 teacher | AdamW, cosine | 1e-4 (ince ayar) | ~32k kare | bf16 |
| T4 DMD2 | AdamW | öğrenci 2e-5, sahte skor 5e-5 | ~16k kare | bf16 |

**Süre tahmini:** model küçük olduğu için süreyi veri okuma ve decoder GAN belirler; güvenilir rakamı T1'in ilk koşusunda adım/sn ölçerek veririm. Şimdiden tahmin vermiyorum.

## 5. Paketler (onay gerekiyor)

| Paket | Neden | Lisans |
|---|---|---|
| `torch` (CUDA 12.x derlemesi) — `training\.venv` | GPU eğitimi (mevcut venv CPU torch) | BSD-3 |
| `numpy`, `ema-lightning==1.0.1`, `normalizer-tr` | Model tanımları + aynı frontend | Apache-2.0 |
| ~~`soundfile`~~ | gerekmiyor (veri PCM16) | — |
| `tensorboard` (opsiyonel) | Eğitim eğrileri | Apache-2.0 |

Başka paket eklenmez; eklenmesi gerekirse sorulur.

## 6. Riskler

| Risk | Etki | Önlem |
|---|---|---|
| Encoder'ın latent uzayı mevcut akustik modelinkiyle uyuşmaz | T3 başlangıç avantajı azalır | Moment eşleme; gerekirse T3 daha uzun |
| Decoder erkek tınısını tam veremez | Tını kadınsı / metalik | T1b GAN ince ayarı, kapıda dinleme |
| CTC hizalama hataları | Kelime kaymaları, yanlış tempo | Val CER + örnek kontrol; düşük güvenli kayıtları ayıklama |
| DMD2 kararsızlığı / mod çöküşü | Öğrenci kalitesi düşer | İki zaman ölçeği, küçük LR, sık dinleme kapısı; teacher her zaman yedek |
| 24 kHz kaynak | Çıktı 12 kHz bant | Kullanıcı kabul etti |
| Windows'ta uzun eğitim | Kesinti | Düzenli checkpoint + kaldığı yerden devam |

## 7. Veri incelemesi (2026-10-07, salt okuma)

**`<datasets>\100HRSKARAY` — kullanıcının verisi, ticari serbest (kullanıcı beyanı)**

| Özellik | Bulgu |
|---|---|
| Ses | 96.141 wav, PCM16 mono 24 kHz (200 rastgele örnekte tümü); ortalama 3.8 sn → ~101 saat (tahmin) |
| Metin | `metadata.csv` 96.136 satır, `ID.wav\| metin` (dikey çizgiden sonra boşluk); 5 wav'ın metni yok → raporlanıp dışlanır |
| Diğer | `train.json` / `val.json`: [metin, token id listesi] — önceki bir eğitimden; metin kaynağı olarak `metadata.csv` kullanılır |
| İçerik | Ağırlıkla sözlük tanımları, kısa, küçük harf; okuma ritmi model tarafından öğrenilir (günlük konuşma tonlaması sınırlı olabilir) |
| Şüpheli satırlar | ör. "sayesan", "i-v-d" — transkripsiyon hatası olabilir; T2 CTC güven skoruyla raporlanır, onaysız silinmez |

`soundfile` gerekmez (PCM16 → `wave` + numpy).

## 8. Kararlar (2026-10-07, kullanıcı)

| Konu | Karar |
|---|---|
| Veri | Yalnız `100HRSKARAY` (kullanıcının sesi) |
| Paketler | §5 onaylı: CUDA torch (cu130, sürücü CUDA 13.1), ema-lightning, normalizer-tr, tensorboard |
| Değerlendirme | Kendi CTC'miz (CER) yeterli; Whisper yok |
| Kapılar | Her faz sonu + **T3 ve T4 içinde ara checkpoint örnekleri** |
| Val/test | Yeni, tohumlu rastgele %1 val + %1 test (`val.json` kullanılmaz) |
| Örnekleme hızı | ~~48 kHz'e çıkarılıp saklanır~~ → **24 kHz'te eğitim**, erkek decoder hop 960, ses kopyası yok (kullanıcı, aynı gün) |
