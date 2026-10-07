# Beyza ses modeli — durum

Onay (kullanıcı, 2026-10-07): sentetik ~100 sa alt küme + gerçek 2.08 sa; decoder yalnız gerçek veri; teacher karışık → gerçek
ince ayar; DMD2 4 adım; ifade klipleri (emoji/etiket → Beyza'nın gerçek kayıtları), eşlenmeyen emoji/etiket sessizce atılır.

## Veri

| Kaynak | Yol | Durum |
|---|---|---|
| Gerçek | `<datasets>\Beyza_real` → `data\beyza` | T0: 606 kayıt, 2.08 sa, 24 kHz; 519'u >10 sn (bölünecek) |
| Sentetik | `<datasets>\audiosINTPBeyza` (130.484 wav, 48 kHz, ~492 sa) → `data\beyza_synth_src` | ~100 sa tohumlu alt küme, 24 kHz'e indirme |
| İfade klipleri | `<recordings>\BEYZA_EMOTION` | 60 klip, 29 kategori, 59.5 sn, 24 kHz |

## Plan

1. Sentetik alt küme (s0) → T0 → birleştirme (r_/s_ önekleri; val/test: gerçek 10/10, sentetik 200/200)
2. CTC ince ayar (erkek CTC'den, Beyza gerçek+sentetik) → hizalama → uzun kayıtları bölme (≤10 sn)
3. T1a encoder (erkek encoder'dan, karışık veri, orijinal kadın decoder dondurulmuş)
4. T1b decoder 24 kHz (orijinal kadın decoder'dan, **yalnız gerçek veri**)
5. Latentler → T3 teacher (orijinal kadın `ema.pt`'den, karışık) → gerçek-veri ince ayarı
6. T4 DMD2 → T5 değerlendirme (gerçek test) → `models\beyza` → servis `speaker: beyza`
7. İfade klipleri: C# frontend/Playhead

## Zaman çizelgesi

| Olay | Sonuç |
|---|---|
| T0 gerçek | 606 / 2.08 sa; erkek CTC ile hizalama skoru medyan −1.30 (erkekte −0.08) → CTC ince ayarı gerekli |
| s0 bitti | 26.500 dosya, 98.74 sa, atlanan 0 (12 işçi, ~30 dk) |
| T0 sentetik | 26.494 kullanılabilir, 89.79 sa (6 çok kısa reddedildi), bant hepsi ≥10 kHz |
| m0 birleştirme | `data\beyza_all`: 27.100 kayıt (r 606 / s 26.494); val/test r 10/10, s 200/200 |
| CTC ince ayar | erkek ctc.pt'den, 4 epoch, lr 3e-4; val CER (10 r + 200 s) %13.55 → 11.56 → 10.54 → 10.37 → `checkpoints\beyza\ctc\ctc.pt` |
| Hizalama | 27.097 / 27.100 (3 başarısız); skor medyan: gerçek **−0.273** (erkek CTC ile −1.30), sentetik −0.141; tüm veri p1 −0.952, p0.5 −1.233 |
| Bölme (t2_split) | 49.519 parça (≤10 sn): gerçek 1.136 / 2.08 sa, sentetik 48.383 / 89.58 sa; 71 parça atıldı |
| T1a bitti | erkek encoder'dan, `data\beyza_all\manifest.jsonl`, 10k adım, batch 16 × 2.88 sn; val mel 2.5k 0.741 → 5k 0.715 → 7.5k 0.709 → 10k 0.698 |
| T1b başladı | orijinal kadın decoder'dan, `data\beyza_all\manifest_real.jsonl` (yalnız gerçek, 606; val 10), 15k adım, 16 × 0.96 sn; 2.5k val mel 0.591 |
| Latentler | `data\beyza_train\` (= manifest_split / durations_split kopyası) 49.519 parça, 8.25 M kare |
| T3 teacher (karışık) | orijinal `ema.pt`'den, 10k adım, min-score −0.952; eğitim 48.199 / val 376 |
| ‼️ T3 log CER yanlış | Judge `--decoder24` verilmezse **erkek** decoder24'e düşüyor → log CER %32–41 anlamsız. Gerçek latentler: erkek decoder24 %54.2, kadın 48k %16.1 (taban). Teacher kadın 48k ile: 7.5k **%6.2**, 10k **%5.85**. Örnekler: `checkpoints\beyza\teacher\samples_female48\` |
| T3 gerçek ince ayar | `t3_teacher.py --init teacher.pt --source r`, 2k adım, lr 2e-5, 1.092 eğitim / 19 val; snap_500/1000/1500 + teacher.pt (2k) → `checkpoints\beyza\teacher_real` |
| T3 karşılaştırma (40 gerçek val+test cümlesi, kadın 48k decoder, Beyza CTC) | karışık teacher **%5.60**; ince ayar 500 %5.88, 1k %6.56, 1.5k %6.86, 2k %8.50 → ince ayar anlaşılırlığı tekdüze düşürüyor (2 sa veride ezber); seçim kulak testine bağlı |
| Teacher seçimi | **A: karışık teacher** (kullanıcı) |
| T1b bitti | 15k; val mel 2.5k 0.591, 5k 0.612, 7.5k 0.586, 10k 0.587, 12.5k 0.579, 15k **0.577** → `checkpoints\beyza\decoder\decoder24.pt` |
| T4 DMD2 (sürüyor) | karışık teacher'dan, 8k adım hedef, her 500'de CER + snap; 8 val (kadın 48k): 0 %9.62, 0.5k 6.33, 1k 5.85, 1.5k 5.60, 2k 5.36, 2.5k 5.24, 3k 5.60, 3.5k 5.12, 4k **4.87** |
| 40 gerçek cümle, Beyza decoder24 + Beyza CTC | kayıtların kendisi %25.81 (taban); teacher 16 adım %4.59; öğrenci 4 adım 2.5k %4.21, 3.5k %4.21, 4k **%3.96**. Dinleme: `checkpoints\beyza\listen\` |
| 40 cümle devam | 4.5k %4.18, 5k %4.13, 5.5k %3.91, 6k **%3.80** (8 val ölçümü 4.5k–6k %5.5–6.3: gürültülü, kullanılmadı) |
| T4 durduruldu | 6.3k'da (kullanıcı); `student\ema.pt` = snap_6000 |
| T5 (`--ctc`/`--source r` eklendi) | 21 gerçek test cümlesi, noktalamasız: kayıtlar %25.43, teacher 16 adım %2.79, öğrenci 4 adım **%4.04**; GPU RTF teacher 0.023, öğrenci 0.0054 → `checkpoints\beyza\t5_report.json`, `samples\test\` |
| Dışa aktarım | `models\beyza\` (24 kHz, hop 960) + `expressions\` 60 wav; appsettings `beyza` eklendi. Servis yeniden derleme + başlatma onay bekliyor |
| Servis (kullanıcı onayı) | eski servis durduruldu, derlendi (0 uyarı/0 hata), başlatıldı; `/voices` → beyza, female, male. POST /tts testleri `test-output\beyza_test1..4.wav`: düz cümle 1.59 sn (ilk istek) / 5.28 sn ses; 😂 😢, `<laugh>` `<ayy>` klipleri araya giriyor; 🚀 ve `<xyz>` sessizce atlandı. CTC dökümü düz cümlede doğru. WS /speak (stream) klipli metinle henüz denenmedi |

## BeyzaR (yalnız gerçek veri, ayrı konuşmacı)

Sebep: kullanıcı testi — Beyza "biraz sentetik" (teacher/öğrenci verisinin yalnız %2'si gerçek). Decoder/encoder/CTC Beyza'nınki.

| Olay | Sonuç |
|---|---|
| Veri | `data\beyzaR_train` (source=r): 1.096 train / 19 val / 21 test; eğitimde 1.092 |
| Teacher | orijinal `ema.pt`'den, 3k adım, her 500'de snap; 8 val CER: 500 %24.6, 1k %14.1, 1.5k **%11.5**, 2k %12.6 |
| Seçim | **snap_1500** — kullanıcı: "örnekler çok iyi" |
| DMD2 | snap_1500'den; 8 val CER: 0 %25.1, 500 %14.7, 1k %11.8, 1.5k **%11.1**, 2k %11.8, 2.5k %15.2, 3k %16.4 (ezber) → ~3.2k'da durduruldu (kullanıcı) |
| Servis | **BeyzaR = öğrenci 1.500** (`student\ema_1500.pt` → `models\beyzaR`); kullanıcı: "BeyzaR, sentetik Beyza'dan çok daha iyi" |
| İyileştirme notu | `<recordings>` incelendi: `Beyza_real` bunun 606 segmenti (metinler Whisper çıktısı gibi); kullanılmamış yalnız 4-30-001…004.m4a (~15.5 dk). Kullanıcı: uğraşma. Whisper kurulumu ağ yok (DNS) → yapılamadı |

## Kararlar

| Konu | Karar | Neden |
|---|---|---|
| T3/T4 `--min-score` | **−0.952** (kullanıcı, 2026-10-07) | Beyza verisinin alt %1'i; erkek eşiği −0.332 gerçek verinin >%25'ini düşürürdü |
| 30HRS_APPLIO_SENTETIK_EMOJI | Karay sentetik (kullanıcı); Beyza'da kullanılmıyor | Etiketli sözsüz sesler; Karay için sonraki aşama |
