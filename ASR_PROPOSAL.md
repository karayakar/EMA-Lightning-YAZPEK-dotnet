# Türkçe ASR Önerisi (ASR_PROPOSAL)

Durum: **ÖNERİ — PARK.** Kod yok. Kullanıcı ~20 erkek + ~20 kadın konuşmacı verisini temin edince başlanır.
Tarih: 2026-10-07. Kardeş işler: `TRAINING_PROPOSAL.md` (erkek TTS, tamam), `MULTI_SPEAKER_PROPOSAL.md` (PARK).
Not: AKATTS_2026 projesinde de Conformer-CTC ASR planı var; bu öneri ondan **bağımsız** yazıldı — birleştirme ya da ayrı tutma kararı kullanıcıda (§9-1).

## 0. Hedef ve şartlar

| Şart | Karşılığı |
|---|---|
| Türkçe konuşma → metin | Konuşmacıdan bağımsız (eğitimde görülmemiş seslerde de çalışan) model |
| Hız | CPU'da gerçek zamandan çok hızlı; canlı kullanımda düşük gecikme (hedefler §6'da, ölçülerek) |
| Dağıtım | TTS ile aynı: ONNX + .NET 8, CPU, HTTP + WebSocket |
| Veri | Kullanıcı temin eder (~40 konuşmacı); dış model/veri yok |
| Ticari kullanım | Yalnız kendi eğittiğimiz modeller; ek paketler lisansıyla onaya sunulur |

## 1. Bu geceki deneyimden gelenler

`training\t2_ctc.py` zaten küçük bir CTC ASR'dir: tek konuşmacı, 100 saat, 6 epoch → val CER %3.6. Yeniden kullanılacaklar: veri okuma/kırpma (`common\audio.py`, `data.py`), log-mel, CTC eğitim/greedy çözme, normalizer-tr (eğitim metinleri okunuş biçimine çevrilir), değerlendirme düzeni, ONNX + C# servis iskeleti.

## 2. Mimari

| Bileşen | Öneri | Neden |
|---|---|---|
| Giriş | 16 kHz log-mel 80 bant, 10 ms adım (telefon hedefi varsa 8 kHz bant sınırlaması eğitimde de uygulanır) | ASR'de standart; 24/48 kHz gereksiz |
| Ön işleme | 2× konvolüsyon alt örnekleme → 40 ms kare (25 Hz) | Hız; Türkçe harf hızı (~14/sn) için yeterli |
| Kodlayıcı | **Conformer** (konvolüsyon + öz dikkat), ~12–16 blok, d≈256 → ~15–30 M parametre (iki boy denenir) | Konuşmacıdan bağımsızlıkta BiLSTM'den güçlü |
| Akış | **Chunk'lı dikkat** (sınırlı geleceğe bakış, ör. 320–640 ms) + önbellekli konvolüsyon → canlı kullanım; aynı ağırlıklar tam sesli (offline) modda da çalışır | Tek model iki kullanım |
| Çıkış | **CTC**, harf alfabesi (Türkçe harfler + boşluk + kesme + temel noktalama) | Sondan eklemeli Türkçe için kelime/BPE sözlüğüne bağlı değil; basit ve hızlı |
| Çözme | Greedy (varsayılan) + isteğe bağlı **n-gram dil modeli ile beam search** (§5) | Özel isim / alan terimlerinde hata düşer |

Alternatif (gerekirse sonra): RNN-T/TDT kafası — akışta daha iyi doğruluk, eğitimi daha ağır. İlk sürüm CTC.

## 3. Veri

| Konu | Plan |
|---|---|
| Hazırlık | TTS T0 düzeni: format denetimi, kırpma, seviye, metin → normalizer-tr fallback (okunuş biçimi), ret raporu |
| Bölme | **Konuşmacı bazlı test**: 2 erkek + 2 kadın konuşmacı tamamen eğitim dışı (görülmemiş ses ölçümü); kalanlarda kayıt bazlı %1 val + %1 test |
| Denge | Konuşmacı başına saat raporu; az verili konuşmacılar örneklemede ağırlıklandırılır (oranlar onaya sunulur) |
| Artırma | Hız 0.9/1.0/1.1, SpecAugment, gürültü + oda yankısı (kullanıcının gürültü/oda verisi ya da sentetik), telefon bandı simülasyonu (hedefe göre) |
| Hizalama | Gerekmez (CTC kendisi öğrenir) |
| Sentetik ek (opsiyonel) | Eğitilen TTS sesleriyle metin çeşitliliği artırma — gerçek veriye oranı sınırlı tutulur (sentetik yanlılık riski) |

## 4. Çıktı biçimi: okunuş → yazı (ters normalizasyon, ITN)

Model **okunuş biçimi** üretir ("on dört otuzda bir milyon iki yüz elli bin türk lirası"). Yazı biçimi için kural tabanlı **Türkçe ITN** (C#):

| Sınıf | Örnek |
|---|---|
| Sayı, ondalık, sıra | "yirmi beş virgül beş" → 25,5; "üçüncü" → 3. |
| Tarih, saat | "on beş ekim iki bin yirmi altı" → 15 Ekim 2026; "on dört otuzda" → 14:30'da |
| Para, birim, yüzde | "bin iki yüz türk lirası" → 1.200 TL; "yüzde on beş" → %15; "beş kilogram" → 5 kg |
| Telefon, e-posta, web | normalizer-tr'nin okuduğu biçimlerin tersi |

normalizer-tr bu yönde çalışmaz; ITN ayrı yazılır. Test: normalizer-tr fixture'ları **tersinden** (yazı → okunuş → ITN → yazı) gidiş-dönüş testi olarak kullanılır.
Noktalama ve büyük harf: ilk sürümde CTC'nin ürettiği temel noktalama + cümle başı büyük harf kuralı; ayrı küçük bir noktalama modeli sonraki adım (§8).

## 5. Dil modeli (opsiyonel)

- Kullanıcının metinlerinden (ve izin verilen metin derlemlerinden) **karakter/kelime n-gram** dil modeli.
- Seçenek A: kendi basit n-gram + beam search (C#, paket yok). Seçenek B: KenLM benzeri bir araç (paket + lisans onayı gerekir).
- Alan sözlüğü / sıcak kelimeler (hotword) desteği beam search'e eklenebilir.

## 6. Hedefler (ölçülecek; şimdiden rakam iddia edilmez)

| Ölçüt | Nasıl ölçülür |
|---|---|
| WER / CER | Görülmüş konuşmacılar test seti + **görülmemiş 4 konuşmacı** ayrı ayrı; konuşmacı başına tablo |
| CPU hızı (RTF) | Offline ve akış modlarında, servis üzerinden uçtan uca |
| Akış gecikmesi | Kelimenin söylenmesi ile metnin gelmesi arası (ilk kısmi sonuç, son sonuç) |
| ITN doğruluğu | Gidiş-dönüş testi (§4) |

## 7. Fazlar

| Faz | İş | Kapı |
|---|---|---|
| **A0 Veri** | Manifest, konuşmacı bazlı bölme, ret ve denge raporu | Rapor onayı |
| **A1 Taban model** | Conformer-CTC (offline, tam ses), artırmalı eğitim; iki boy (küçük/orta) | Konuşmacı başına WER/CER, görülmemiş konuşmacılar |
| **A2 Akış** | Chunk'lı dikkat ile ince ayar; gecikme/doğruluk ölçümü | Gecikme + WER tablosu |
| **A3 ITN** | Kural tabanlı Türkçe ITN (C#) + gidiş-dönüş testleri | Test geçme oranı |
| **A4 Dil modeli (ops.)** | n-gram + beam search; WER farkı ölçülür | Kullanıcı kararı |
| **A5 Export + .NET** | ONNX (offline + akış durumlu), C# servis: `POST /asr` (dosya), `WS /listen` (canlı, kısmi/son sonuç), test sayfası | Uçtan uca CPU RTF + gecikme |

Her fazda örnek çıktı tabloları + `ASR_STATUS.md` (TTS gecesindeki düzen).

## 8. Kapsam dışı (sonraki adımlar)

Konuşmacı ayrıştırma (kim ne zaman konuştu), dil tespiti, ayrı noktalama/büyük harf modeli, kelime zaman damgaları (CTC hizalamasıyla kolay — istenirse A5'e eklenir), RNN-T kafası.

## 9. Açık sorular

1. **AKATTS_2026 ile ilişki:** bu ASR o projeye mi bağlansın, ayrı mı kalsın?
2. Konuşmacı başına saat, örnekleme hızı, format (wav+metin / parquet / csv), klasörler.
3. Kullanım: canlı akış mı (mikrofon / telefon 8 kHz), dosya transkripsiyonu mu, ikisi mi?
4. Konuşma tipi: okuma mı, serbest konuşma mı; ortam temiz mi gürültülü mü? Gürültü/oda verisi var mı?
5. Çıktı: yazı biçimi (rakam, noktalama, büyük harf) mi, okunuş biçimi mi, ikisi mi?
6. Görülmemiş-konuşmacı testi için ayrılacak 4 konuşmacıyı kullanıcı mı seçer, rastgele mi?
7. Proje klasörü: bu klasör (`<repo>`) altında mı, ayrı bir proje klasörü mü?
