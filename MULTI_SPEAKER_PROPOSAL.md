# EMA Lightning — Çok Konuşmacılı Model Önerisi (MULTI_SPEAKER_PROPOSAL)

Durum: **ÖNERİ — PARK.** Kod yok. Kullanıcı 3–4 kadın + 3–4 erkek konuşmacı verisini temin edince başlanır.
Tarih: 2026-10-07. Önceki iş: `TRAINING_PROPOSAL.md` (tek erkek ses, T0–T5, tamamlandı — `TRAINING_STATUS.md`).

## 0. Hedef ve şartlar

| Şart | Karşılığı |
|---|---|
| Tek model, çok ses | Konuşmacı kimliği (speaker embedding) girdisi; istekte `"speaker": "<ad>"` |
| **Hız kaybı yok** | Mimari aynı + 1 tablo okuması ve 2 küçük doğrusal katman; 4 adım DMD2 korunur |
| Veri | Kullanıcı temin eder (3–4 kadın, 3–4 erkek) |
| Konuşmacı profilleri (referans kayıttan stil çıkarma, Supertonic tarzı) | **Bu önerinin dışında, sonraki aşama** (§7) |

## 1. Mimari değişiklik (küçük)

Kaynak: `ema_lightning\model.py` (`Acoustic`). Mevcut koşullama: `c = t_embed(t)` → `ada_shared(c)` (6 bloğun modülasyonu) ve `ada_out(c)`.

| Yer | Bugün | Öneri |
|---|---|---|
| Konuşmacı tablosu | yok | `speaker = nn.Embedding(S, 224)`, **sıfıra yakın başlatma** (başlangıçta model bugünkü gibi davranır) |
| DiT koşulu | `c = t_embed(t)` | `c = t_embed(t) + speaker_proj(e_s)` → ortak modülasyon üzerinden 6 bloğu ve çıkış modülasyonunu etkiler (tını, perde) |
| Süre tahmini | `chardur(h, mask)` | `chardur(h + speaker_dur(e_s)[:, None], mask)` → konuşmacıya göre tempo. `h`'nin kendisi (aligner girdisi) değişmez |
| Metin kodlayıcı, aligner | — | Değişmez |
| Decoder | ses başına ayrı (kadın 48 kHz, erkek 24 kHz) | **Tek ortak decoder**, konuşmacı girdisi yok (tını latentte); tüm konuşmacılarla eğitilir |
| CFG | metin koşulu düşürülür | Yalnız metin koşulu düşürülür; konuşmacı her zaman verilir (rehberlik metin üzerinde, ses kimliği sabit) |

Parametre artışı: S × 224 + 2 × (224 × 224) ≈ 0.1 M → toplam ~5.66 M (bugün 5.56 M).

**Ses karışımı (yan ürün):** iki konuşmacı vektörünün doğrusal karışımı (`0.7·e_a + 0.3·e_b`) ara sesler üretebilir. Kalitesi eğitimden sonra denenir; garanti edilmez.

## 2. ONNX ve C# değişiklikleri

| Katman | Değişiklik |
|---|---|
| `text.onnx` | + girdi `speaker` int64 [B] (süre tahmini için) |
| `sound.onnx` | + girdi `speaker` int64 [B] |
| `decoder.onnx` | değişmez (ortak) |
| `ema_meta.json` | + `"speakers": ["ayse", "mehmet", …]` (sıra = id) |
| C# Engine | `Piece.Speaker`; Plan/Think `speaker` tensörünü batch'ler (farklı konuşmacılar aynı batch'te karışık olabilir) |
| C# Ema / Server | Tek `Ema` (tek model), istekte `speaker` adı → id; `GET /voices` meta'dan listeler. Mevcut çoklu model seti desteği kalır (eski kadın/erkek setleri yan yana çalışabilir) |
| Arayüz | "Speaker" listesi meta'dan dolar; karışım (opsiyonel, §1) için ikinci konuşmacı + oran |

## 3. Örnekleme hızı

Tüm veri tek hıza çekilir; ortak decoder o hızda olur:

- Herhangi bir konuşmacı 24 kHz ise → **24 kHz decoder** (hop 960; bugünkü erkek decoder'ından başlatılır).
- Hepsi ≥ 44.1 kHz ise → 48 kHz decoder (orijinal `decoder.pt`'den başlatılabilir) — **veri gelince karar** (§8).

## 4. Fazlar (TRAINING_PROPOSAL ile aynı akış, çok konuşmacılı)

| Faz | İş | Not |
|---|---|---|
| **M0 Veri** | Konuşmacı başına manifest (id, ad, metin, kırpma, kazanç), ortak hız, konuşmacı başına val/test %1 | `t0_prepare.py` genişler: `--speaker` / çoklu kaynak |
| M0b (opsiyonel) | **Orijinal kadın sesi** konuşmacılardan biri olacaksa: mevcut kadın modeliyle metinlerden sentetik veri üretilir | Kullanıcı kararı (§8) |
| **M1 Encoder + ortak decoder** | Encoder tüm konuşmacılarla (latent uzayı tüm tınıları kapsasın); decoder GAN ince ayar, tüm konuşmacılar | Konuşmacı başına yeniden sentez dinleme örnekleri |
| **M2 Hizalama** | Ortak CTC (tüm veri) → kelime süreleri; konuşmacı başına skor raporu, alt %1 eleme | Bu gecenin `t2_ctc.py` / `t2_align.py`'si |
| **M3 Teacher** | Konuşmacı tablosu eklenmiş Acoustic; bugünkü erkek teacher ağırlıklarından başlar; flow matching + süre | Konuşmacı başına CER + F0/tempo ara örnekleri |
| **M4 DMD2** | 4 adım, CFG 4.0 gömülü; öğrenci/teacher/sahte skor konuşmacı koşullu | Konuşmacı başına ara örnekler |
| **M5 Değerlendirme + export + C#** | Konuşmacı başına CER (noktalamasız), F0, tempo, CPU RTF; ONNX (speaker girdili); C# + arayüz | Hız kontrolü: RTF bugünküyle aynı olmalı |

Kontrol kapıları: her faz sonunda konuşmacı başına örnek + metrik (`TRAINING_STATUS` düzeninde).

## 5. Veri gereksinimi (tahmin, ölçülmedi)

| Konuşmacı başına | Beklenti |
|---|---|
| ≥ 10 saat | Tek konuşmacı kalitesine yakın |
| 2–10 saat | İyi; ortak model diğer konuşmacılardan öğrenir |
| < 1 saat | Tını tutar ama telaffuz/tempo zayıflayabilir |

Konuşmacılar arası saat dengesizliği varsa eğitimde konuşmacı başına örnekleme ağırlığı (az verisi olan daha sık) uygulanır — veri gelince oranlar raporlanıp onaya sunulur.

## 6. Riskler

| Risk | Önlem |
|---|---|
| Ortak decoder bir sesi bozar | Konuşmacı başına yeniden sentez metrikleri; gerekirse konuşmacı koşullu decoder (sonraki adım) |
| Az verili konuşmacıda tını kayması | Örnekleme ağırlığı, konuşmacı başına F0/spektral denetim |
| Farklı kayıt koşulları (mikrofon, oda) konuşmacı kimliğiyle karışır | T0'da bant/gürültü raporu; seviye eşitleme |
| Karışım sesleri doğal olmayabilir | Yan ürün; garanti değil |

## 7. Sonraki aşama (bu önerinin dışında): konuşmacı profilleri

Referans kayıttan **stil vektörü** çıkaran bir kodlayıcı (Supertonic tarzı "voice style"), profilin düzenlenmesi/karıştırılması. Konuşmacı tablosu (§1) bunun ön adımıdır: tablo yerine kodlayıcının ürettiği vektör aynı yere bağlanır. Ayrı öneri yazılacak; Supertonic'in ONNX girdileri ve stil vektörü yapısı o aşamada incelenecek.

## 8. Açık sorular (veri gelince)

1. Konuşmacı adları (API `speaker` değerleri), cinsiyet etiketleri.
2. Her konuşmacı: klasör, format (wav+metin / parquet / csv), saat, örnekleme hızı.
3. Bugünkü 100 saatlik erkek veri (`100HRSKARAY`) konuşmacılardan biri olsun mu?
4. Orijinal kadın sesi (EMA Lightning) sentetik veriyle konuşmacılardan biri olsun mu (M0b)?
5. Ortak decoder hızı (§3) — veri hızlarına göre önerilecek.
6. Eski tek-ses setleri (`models\`, `models\male\`) çok konuşmacılı model gelince kalsın mı?
