// normalizer-tr C# portunu Rust deposundaki beklenen çıktılarla karşılaştırır.
// Kaynaklar: tests/fixtures/current-contract.json (+ benches/corpus.json, intent-corpus.json),
// tests/fixtures/financial-*.txt ve tests/*.rs içindeki sabit (girdi, beklenen) çiftleri.
// Kullanım: dotnet run -- [normalizer-tr klasörü]   (verilmezse yukarı doğru github\normalizer-tr aranır)
using System.Text;
using System.Text.Json.Nodes;
using NormalizerTr;

var root = args.Length > 0 ? args[0] : FindRepo();
Console.OutputEncoding = Encoding.UTF8;
Console.WriteLine($"normalizer-tr: {root}");

var normalizer = new Normalizer();
var failures = new List<string>();
var passed = 0;

void Check(bool ok, string label)
{
    if (ok)
    {
        passed++;
    }
    else
    {
        failures.Add(label);
    }
}

NormalizeResult? Run(string input, AmbiguityPolicy policy, HintKind? hint = null)
{
    var options = new NormalizeOptions { AmbiguityPolicy = policy };
    if (hint is { } kind)
    {
        options.Hints.Add(new Hint(new SourceRange(0, Encoding.UTF8.GetByteCount(input)), kind));
    }
    return normalizer.Normalize(input, options);
}

string? ErrorOf(Func<object?> action)
{
    try
    {
        action();
        return null;
    }
    catch (NormalizeException e)
    {
        return e.DebugName;
    }
}

// 1) Contract oracle (preserve + reject)
{
    var corpus = new List<JsonNode>();
    corpus.AddRange(JsonNode.Parse(File.ReadAllText(Path.Combine(root, "benches", "corpus.json")))!.AsArray()!);
    corpus.AddRange(JsonNode.Parse(File.ReadAllText(Path.Combine(root, "benches", "intent-corpus.json")))!.AsArray()!);
    var expected = JsonNode.Parse(File.ReadAllText(Path.Combine(root, "tests", "fixtures", "current-contract.json")))!;
    foreach (var item in corpus)
    {
        foreach (var reject in new[] { false, true })
        {
            var options = new NormalizeOptions
            {
                AmbiguityPolicy = reject ? AmbiguityPolicy.Reject : AmbiguityPolicy.Preserve,
            };
            if (item["hint"] is { } hint)
            {
                var kind = hint["kind"]!.GetValue<string>() switch
                {
                    "cardinal" => HintKind.Cardinal,
                    "digits" => HintKind.Digits,
                    "date" => HintKind.Date,
                    "time" => HintKind.Time,
                    "ordinal" => HintKind.Ordinal,
                    "roman" => HintKind.Roman,
                    "range" => HintKind.Range,
                    "telephone" => HintKind.Telephone,
                    _ => HintKind.Electronic,
                };
                options.Hints.Add(new Hint(
                    new SourceRange(hint["start"]!.GetValue<int>(), hint["end"]!.GetValue<int>()), kind));
            }
            JsonNode actual;
            try
            {
                var result = normalizer.Normalize(item["text"]!.GetValue<string>(), options);
                if (result.Fallbacks.Count != 0 || result.FallbackUsed)
                {
                    failures.Add($"contract {item["id"]}: unexpected fallbacks");
                }
                actual = new JsonObject
                {
                    ["result"] = new JsonObject
                    {
                        ["complete"] = result.Complete,
                        ["issues"] = IssuesJson(result.Issues),
                        ["locale"] = result.Locale,
                        ["normalized_text"] = result.NormalizedText,
                        ["segments"] = new JsonArray(result.Segments.Select(s => (JsonNode)new JsonObject
                        {
                            ["kind"] = s.Kind.ToString(),
                            ["range"] = RangeJson(s.Range),
                            ["text"] = s.Text,
                        }).ToArray()),
                    },
                };
            }
            catch (NormalizeException e) when (e.Kind == NormalizeErrorKind.Unresolved)
            {
                actual = new JsonObject { ["error"] = "unresolved", ["issues"] = IssuesJson(e.Issues) };
            }
            catch (NormalizeException e)
            {
                actual = new JsonObject { ["error"] = e.DebugName };
            }
            var key = $"{item["id"]!.GetValue<string>()}:{(reject ? "reject" : "preserve")}";
            var want = expected["outcomes"]![key];
            var same = Same(actual, want);
            Check(same, $"contract {key}\n    expected: {want?.ToJsonString()}\n    actual:   {actual.ToJsonString()}");
        }
    }
}

// 2) Financial baseline (3 politika)
{
    var input = File.ReadAllText(Path.Combine(root, "tests", "fixtures", "financial-input.txt")).TrimEnd('\r', '\n');
    var output = File.ReadAllText(Path.Combine(root, "tests", "fixtures", "financial-output.txt")).TrimEnd('\r', '\n');
    foreach (var policy in new[] { AmbiguityPolicy.Preserve, AmbiguityPolicy.Reject, AmbiguityPolicy.Fallback })
    {
        var result = Run(input, policy)!;
        var transformed = string.Join(";", result.Segments.Where(s => s.Kind != SegmentKind.Verbatim)
            .Select(s => $"{s.Kind}:{s.Range.Start}-{s.Range.End}"));
        Check(result.NormalizedText == output && result.Complete && !result.FallbackUsed
              && transformed == "Date:0-13;Time:19-27;Percent:51-60;Abbreviation:70-73;Money:80-95",
            $"financial {policy}: {result.NormalizedText} | {transformed}");
    }
}

// 3) Sabit beklenen çıktılar (tests/*.rs)
void Expect(string input, string expected, AmbiguityPolicy policy = AmbiguityPolicy.Preserve,
    HintKind? hint = null, bool complete = true)
{
    try
    {
        var result = Run(input, policy, hint)!;
        Check(result.NormalizedText == expected && result.Complete == complete,
            $"{policy} {hint} \"{input}\"\n    expected: {expected}\n    actual:   {result.NormalizedText} (complete={result.Complete})");
    }
    catch (NormalizeException e)
    {
        failures.Add($"{policy} {hint} \"{input}\": {e.DebugName}");
    }
}

void ExpectError(string input, string error, AmbiguityPolicy policy = AmbiguityPolicy.Preserve,
    HintKind? hint = null)
{
    var actual = ErrorOf(() => Run(input, policy, hint));
    Check(actual == error, $"{policy} \"{(input.Length > 40 ? input[..40] + "…" : input)}\": expected {error}, actual {actual ?? "ok"}");
}

const AmbiguityPolicy F = AmbiguityPolicy.Fallback;

// tests/fallback.rs
foreach (var (input, expected) in new[]
         {
             ("1.234", "bin iki yüz otuz dört"),
             ("00042", "sıfır sıfır sıfır dört iki"),
             ("AB12", "a be bir iki"),
             ("ABC", "a be ce"),
             ("IV", "ı ve"),
             ("Toplam 25.", "Toplam yirmi beş."),
             ("10-15", "on tire on beş"),
             ("1/2", "bir eğik çizgi iki"),
             ("12:30", "on iki otuz"),
             ("40.03.2026", "kırk Mart iki bin yirmi altı"),
             ("tarih 31.02.2026", "tarih otuz bir Şubat iki bin yirmi altı"),
             ("saat 25:70", "saat yirmi beş yetmiş"),
             ("2026-03-40", "kırk Mart iki bin yirmi altı"),
             ("tarih 31.13.2026", "tarih otuz bir nokta on üç nokta iki bin yirmi altı"),
             ("25,123 TL", "yirmi beş virgül bir iki üç Türk lirası"),
             ("1.234'te", "bin iki yüz otuz dörtte"),
             ("1.234'dan", "bin iki yüz otuz dört kesme dan"),
             ("TR00 0006 1005 1978 6457 8413 26",
                 "te re sıfır sıfır sıfır sıfır sıfır altı bir sıfır sıfır beş bir dokuz yedi sekiz altı dört beş yedi sekiz dört bir üç iki altı"),
             ("AB0012", "a be sıfır sıfır bir iki"),
             ("προ Hello q́", "προ Hello q́"),
             ("hello🙂world", "hello gülümseyen yüz world"),
             ("🫠", "unikod u artı bir fe a e sıfır"),
             ("ö🙂!", "ö gülümseyen yüz!"),
             ("Söz: (merhaba), \"evet\"!\r\n", "Söz: (merhaba), \"evet\"!\r\n"),
             ("π", "pi"),
         })
{
    Expect(input, expected, F);
}
foreach (var (input, cued) in new[] { ("12:00", "saat 12:00"), ("09:30'da", "saat 09:30'da"), ("14.03.2026'da", "tarih 14.03.2026'da") })
{
    var ordinary = Run(cued, AmbiguityPolicy.Preserve)!.NormalizedText;
    Expect(input, ordinary[(ordinary.IndexOf(' ') + 1)..], F);
}
foreach (var input in new[] { "25 TL", "4,25", "saat 12:05", "%37,5'lik", "II. Dünya Savaşı", "0850 222 33 44", "https://ornek.com/a@b?x=2" })
{
    var primary = Run(input, AmbiguityPolicy.Preserve)!;
    var fallback = Run(input, F)!;
    Check(primary.Complete && fallback.NormalizedText == primary.NormalizedText && !fallback.FallbackUsed,
        $"normal reading wins \"{input}\"");
}
Expect("IV", "dört", F, HintKind.Roman);
foreach (var input in new[] { "https://name:pw@ornek.com/x%ZZ", "AB0012", "Ж12", "abc_0002" })
{
    var result = Run(input, F)!;
    Check(result.Complete && result.Fallbacks.Count == 1 && result.Segments.Count == 1
          && result.Fallbacks[0].Range == new SourceRange(0, Encoding.UTF8.GetByteCount(input))
          && !result.NormalizedText.Contains("%ZZ"), $"single fallback claim \"{input}\"");
}
{
    var result = Run("hello🙂world", F)!;
    Check(result.Fallbacks[0].Range == new SourceRange(5, 9), "hello🙂world fallback range");
    result = Run("ö🙂!", F)!;
    Check(result.Fallbacks[0].Range == new SourceRange(3, 7), "ö🙂! fallback range");
    result = Run("🙂", F)!;
    var d = result.Fallbacks[0];
    Check(d.Range == new SourceRange(0, 4) && d.AttemptedClass == FallbackClass.Symbol
          && d.Reason == FallbackReason.UnhandledSymbol && d.OriginalCategory == null
          && d.Strategy == FallbackStrategy.Literal, "🙂 diagnostic");
}
foreach (var input in new[] { "→", "\\", "|", "✓", "<3", ":D", "***", "ö🙂\r\nABC", "👨‍👩‍👧", "🙂️", ":D́", "q́🙂", "́", "a​b", "İ12" })
{
    var result = Run(input, F)!;
    Check(result.Complete && result.FallbackUsed && result.NormalizedText.Length > 0, $"symbol coverage \"{input}\"");
}
foreach (var input in new[] { "", " ", "a‮12", "a‏b" })
{
    ExpectError(input, "InvalidInput", F);
}
{
    var options = new NormalizeOptions { AmbiguityPolicy = F };
    options.Hints.Add(new Hint(new SourceRange(0, 1), HintKind.Cardinal));
    Check(ErrorOf(() => normalizer.Normalize("3 + 4", options)) == "InvalidHint", "3 + 4 invalid hint");
    var control = new WorkControl();
    control.Cancel();
    Check(ErrorOf(() => normalizer.NormalizeControlled("🙂", options, control)) == "Cancelled", "cancelled");
}
Expect("25 TL; 1.234", "yirmi beş Türk lirası; 1.234", complete: false);
ExpectError(new string('a', NormalizerLimits.MaxInputBytes + 1), "LimitExceeded(Input)", F);
ExpectError(string.Concat(Enumerable.Repeat("🙂 ", 4097)), "LimitExceeded(Candidates)", F);
ExpectError(string.Concat(Enumerable.Repeat("🫠", NormalizerLimits.MaxInputBytes / 4)), "LimitExceeded(Result)", F);

// tests/classes.rs
foreach (var (input, expected) in new[]
         {
             ("0", "sıfır"), ("100", "yüz"), ("1000", "bin"), ("1001", "bin bir"), ("101000", "yüz bir bin"),
             ("1000000", "bir milyon"), ("1000000000", "bir milyar"), ("1000000000000", "bir trilyon"),
             ("1000000000000000", "bir katrilyon"),
             ("999999999999999999",
                 "dokuz yüz doksan dokuz katrilyon dokuz yüz doksan dokuz trilyon dokuz yüz doksan dokuz milyar dokuz yüz doksan dokuz milyon dokuz yüz doksan dokuz bin dokuz yüz doksan dokuz"),
             ("-0", "eksi sıfır"), ("+12", "artı on iki"), ("12,05", "on iki virgül sıfır beş"),
             ("-0,000000001", "eksi sıfır virgül sıfır sıfır sıfır sıfır sıfır sıfır sıfır sıfır bir"),
             ("%12,5", "yüzde on iki virgül beş"),
             ("1.234,50 TL", "bin iki yüz otuz dört Türk lirası elli kuruş"),
             ("-0,50 TL", "eksi sıfır Türk lirası elli kuruş"), ("-0,00 TRY", "eksi sıfır Türk lirası"),
             ("25 ₺", "yirmi beş Türk lirası"), ("₺25,01", "yirmi beş Türk lirası bir kuruş"),
             ("25,5₺", "yirmi beş Türk lirası elli kuruş"), ("+25,00 TL", "artı yirmi beş Türk lirası"),
             ("TL 25", "yirmi beş Türk lirası"), ("1.000.000,09 TRY", "bir milyon Türk lirası dokuz kuruş"),
             ("3'üncü", "üçüncü"), ("6'ncı", "altıncı"), ("1'inci", "birinci"), ("2'nci", "ikinci"),
             ("4'üncü", "dördüncü"), ("4'e", "dörde"), ("4'ü", "dördü"), ("4'ün", "dördün"), ("4'te", "dörtte"),
             ("4'ten", "dörtten"), ("6'ya", "altıya"), ("6'yı", "altıyı"), ("6'nın", "altının"),
             ("100'ün", "yüzün"), ("%4'lük", "yüzde dörtlük"), ("%6'lık", "yüzde altılık"),
             ("%10'luk", "yüzde onluk"), ("%12,5'lik", "yüzde on iki virgül beşlik"),
             ("5 kg", "beş kilogram"), ("3 m", "üç metre"), ("2 g", "iki gram"), ("1 km", "bir kilometre"),
             ("12 cm", "on iki santimetre"), ("10 mm", "on milimetre"), ("2 L", "iki litre"),
             ("0,05 mL", "sıfır virgül sıfır beş mililitre"), ("Dr. Ali", "doktor Ali"), ("TBMM", "te be me me"),
             ("TBMM'ye", "te be me meye"), ("KDV", "katma değer vergisi"), ("TBMM’ye", "te be me meye"),
             ("tarih 01.02.2026", "tarih bir Şubat iki bin yirmi altı"),
             ("tarih: 01.02.2026", "tarih: bir Şubat iki bin yirmi altı"),
             ("TARİH:01.02.2026", "TARİH:bir Şubat iki bin yirmi altı"),
             ("tarih :01.02.2026", "tarih :bir Şubat iki bin yirmi altı"),
             ("tarih : 01.02.2026", "tarih : bir Şubat iki bin yirmi altı"),
             ("TARİH:01.02.2026", "TARİH:bir Şubat iki bin yirmi altı"),
             ("saat: 09:30", "saat: dokuz otuz"), ("saat:09:30", "saat:dokuz otuz"),
             ("tarihi 29.02.2000", "tarihi yirmi dokuz Şubat iki bin"),
             ("tarih 2026-03-14", "tarih on dört Mart iki bin yirmi altı"),
             ("TARİH: 2024-02-29", "TARİH: yirmi dokuz Şubat iki bin yirmi dört"),
             ("saat 12:30", "saat on iki otuz"), ("SAAT 12.30", "SAAT on iki otuz"), ("saat 00:00", "saat sıfır"),
             ("saat 09:00'da", "saat dokuzda"), ("saat 04:00'te", "saat dörtte"),
         })
{
    Expect(input, expected);
}
foreach (var (input, kind, expected) in new[]
         {
             ("1.234", HintKind.Cardinal, "bin iki yüz otuz dört"), ("00042", HintKind.Cardinal, "kırk iki"),
             ("001.234", HintKind.Cardinal, "bin iki yüz otuz dört"), ("-000", HintKind.Cardinal, "eksi sıfır"),
             ("1.234", HintKind.Digits, "bir iki üç dört"), ("12/34", HintKind.Digits, "bir iki üç dört"),
             ("(123)", HintKind.Digits, "bir iki üç"),
             ("+90(532)123", HintKind.Digits, "artı dokuz sıfır beş üç iki bir iki üç"),
             ("12-34", HintKind.Digits, "bir iki üç dört"), ("-12", HintKind.Digits, "bir iki"),
             ("+90 (532) 123 45 67", HintKind.Digits, "artı dokuz sıfır beş üç iki bir iki üç dört beş altı yedi"),
             ("00042", HintKind.Digits, "sıfır sıfır sıfır dört iki"),
             ("+90 532 123 45 67", HintKind.Digits, "artı dokuz sıfır beş üç iki bir iki üç dört beş altı yedi"),
             ("01.02.2026", HintKind.Date, "bir Şubat iki bin yirmi altı"),
             ("2026-03-14", HintKind.Date, "on dört Mart iki bin yirmi altı"),
             ("12:30", HintKind.Time, "on iki otuz"),
             ("25.", HintKind.Ordinal, "yirmi beşinci"), ("1.'nin", HintKind.Ordinal, "birincinin"),
             ("IV", HintKind.Roman, "dört"), ("II.", HintKind.Roman, "ikinci"), ("IV'üncü", HintKind.Roman, "dördüncü"),
             ("IV'üncünün", HintKind.Roman, "dördüncünün"), ("II.'nin", HintKind.Roman, "ikincinin"),
             ("MMMCMXCIX", HintKind.Roman, "üç bin dokuz yüz doksan dokuz"),
             ("10-15", HintKind.Range, "on ila on beş"),
             ("5321234567", HintKind.Telephone, "beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi"),
             ("ornek.com", HintKind.Electronic, "ornek nokta kom"),
         })
{
    Expect(input, expected, hint: kind);
}

// tests/general_coverage.rs
foreach (var (input, expected) in new[]
         {
             ("%3'ten", "yüzde üçten"), ("%3,25'ten", "yüzde üç virgül iki beşten"), ("TL'den", "Türk lirasından"),
             ("25 TL'den", "yirmi beş Türk lirasından"), ("5 kg'dan", "beş kilogramdan"), ("1.'nin", "birincinin"),
             ("4.'ye", "dördüncüye"), ("3'üncü'nün", "üçüncünün"), ("3'üncünün", "üçüncünün"),
             ("1'incinin", "birincinin"), ("6'ncı'ya", "altıncıya"),
             ("25,05 TL'den", "yirmi beş Türk lirası beş kuruştan"), ("25 USD'den", "yirmi beş dolardan"),
             ("25,05 USD'den", "yirmi beş dolar beş sentten"), ("KDV'den", "katma değer vergisinden"),
             ("PTT'ye", "pe te teye"), ("NATO'da", "natoda"), ("IBAN'ın", "ibanın"), ("TBMM'nin", "te be me menin"),
             ("Prof. Ali", "profesör Ali"), ("vb.", "ve benzeri"), ("PTT NATO IBAN", "pe te te nato iban"),
             ("2 mg", "iki miligram"), ("2 µg", "iki mikrogram"), ("2 μg", "iki mikrogram"), ("2 gr", "iki gram"),
             ("2 ml", "iki mililitre"), ("2 lt", "iki litre"), ("2 dk", "iki dakika"), ("2 sn", "iki saniye"),
             ("2 sa", "iki saat"), ("2 m²", "iki metrekare"), ("2 cm²", "iki santimetrekare"),
             ("2 km²", "iki kilometrekare"), ("2 m³", "iki metreküp"),
             ("90 km/sa", "saatte doksan kilometre"), ("90 km/h", "saatte doksan kilometre"),
             ("5 m/s", "saniyede beş metre"), ("-0,05 ml'den", "eksi sıfır virgül sıfır beş mililitreden"),
             ("5 m²'ye", "beş metrekareye"), ("5 m³'e", "beş metrekübe"), ("5 m³'ün", "beş metrekübün"),
             ("5 m³'te", "beş metreküpte"), ("2 sa'ten", "iki saatten"), ("2 sa'te", "iki saatte"),
             ("$40", "kırk dolar"), ("€14,05", "on dört avro beş sent"), ("25 GBP", "yirmi beş sterlin"),
             ("-0,50 USD", "eksi sıfır dolar elli sent"), ("+0,00 EUR", "artı sıfır avro"),
             ("1.234,5 £", "bin iki yüz otuz dört sterlin elli peni"), ("GBP 25", "yirmi beş sterlin"),
             ("25€", "yirmi beş avro"), ("25€'ya", "yirmi beş avroya"), ("25₺'dan", "yirmi beş Türk lirasından"),
             ("₺25", "yirmi beş Türk lirası"),
             ("10-15 Kişi", "on ila on beş Kişi"), ("10-15 kişi", "on ila on beş kişi"),
             ("15-10 adet", "on beş ila on adet"),
             ("1,05-2,50 kg", "bir virgül sıfır beş ila iki virgül beş sıfır kilogram"),
             ("-5--2 gün", "eksi beş ila eksi iki gün"), ("3–4 yaş", "üç ila dört yaş"),
             ("telefon 25 TL", "telefon yirmi beş Türk lirası"), ("telefon 3 adet", "telefon üç adet"),
             ("TELEFON 5321234567", "TELEFON beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi"),
             ("0005 001 02 03", "sıfır sıfır sıfır beş sıfır sıfır bir sıfır iki sıfır üç"),
             ("+90", "artı doksan"), ("+90532", "artı doksan bin beş yüz otuz iki"), ("0 1 2", "sıfır bir iki"),
             ("0 5", "sıfır beş"),
             ("0 0 0 0 0 0 0 0 0 0 0", "sıfır sıfır sıfır sıfır sıfır sıfır sıfır sıfır sıfır sıfır sıfır"),
             ("0,000000000", "sıfır virgül sıfır sıfır sıfır sıfır sıfır sıfır sıfır sıfır sıfır"),
             ("0-123456789 kişi", "sıfır ila yüz yirmi üç milyon dört yüz elli altı bin yedi yüz seksen dokuz kişi"),
             ("0850 222 33 44", "sıfır sekiz yüz elli iki yüz yirmi iki otuz üç kırk dört"),
             ("0850-222-33-44", "sıfır sekiz yüz elli iki yüz yirmi iki otuz üç kırk dört"),
             ("08502223344", "sıfır sekiz yüz elli iki yüz yirmi iki otuz üç kırk dört"),
             ("+90 (532) 123 45 67", "artı doksan beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi"),
             ("0505 001 02 03", "sıfır beş yüz beş sıfır sıfır bir sıfır iki sıfır üç"),
             ("telefon 5321234567", "telefon beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi"),
             ("TR330006100519786457841326",
                 "te re üç üç, sıfır sıfır sıfır altı, bir sıfır sıfır beş, bir dokuz yedi sekiz, altı dört beş yedi, sekiz dört bir üç, iki altı"),
             ("TR33 0006 1005 1978 6457 8413 26",
                 "te re üç üç, sıfır sıfır sıfır altı, bir sıfır sıfır beş, bir dokuz yedi sekiz, altı dört beş yedi, sekiz dört bir üç, iki altı"),
             ("tren ve trafik bugün sakin", "tren ve trafik bugün sakin"),
             ("trafik 25 örnek içerir", "trafik yirmi beş örnek içerir"),
             ("WEB ornek.net", "WEB ornek nokta net"), ("info@ornek.com", "info et ornek nokta kom"),
             ("User01@Ornek.com.tr", "User sıfır bir et Ornek nokta com nokta te re"),
             ("#yapayzeka & araştırma", "hashtag yapayzeka ve araştırma"),
             ("www.ornek.com", "çift ve çift ve çift ve nokta ornek nokta kom"),
             ("web ornek.net", "web ornek nokta net"),
             ("HTTPS://Ornek.com/A00",
                 "ha te te pe es iki nokta eğik çizgi eğik çizgi Ornek nokta kom eğik çizgi A sıfır sıfır"),
             ("https://ornek.com/a@b?x=2",
                 "ha te te pe es iki nokta eğik çizgi eğik çizgi ornek nokta kom eğik çizgi a et b soru işareti x eşittir iki"),
             ("https://ornek.com:8080/a01?x=2&y=%20#bolum",
                 "ha te te pe es iki nokta eğik çizgi eğik çizgi ornek nokta kom iki nokta sekiz sıfır sekiz sıfır eğik çizgi a sıfır bir soru işareti x eşittir iki ve y eşittir yüzde iki sıfır kare bolum"),
             ("(https://ornek.com/a(b)).",
                 "(ha te te pe es iki nokta eğik çizgi eğik çizgi ornek nokta kom eğik çizgi a aç parantez b kapat parantez)."),
             ("II. DÜNYA SAVAŞI", "ikinci DÜNYA SAVAŞI"), ("II. dünya savaşı", "ikinci dünya savaşı"),
             ("II. Dünya Savaşı", "ikinci Dünya Savaşı"), ("XXI. yüzyıl", "yirmi birinci yüzyıl"),
         })
{
    Expect(input, expected);
}
Expect("https://ornek.com/" + new string('a', 4096),
    "ha te te pe es iki nokta eğik çizgi eğik çizgi ornek nokta kom eğik çizgi " + new string('a', 4096));
for (var last = '0'; last <= '9'; last++)
{
    if (last != '6')
    {
        var input = $"TR33000610051978645784132{last}";
        Expect(input, input, complete: false);
    }
}
Expect("AB12 25 TL", "AB12 yirmi beş Türk lirası", complete: false);
foreach (var input in new[]
         {
             "IV", "I.", "Toplam 25.", "10-15", "TR12 0006 1005", "tr330006100519786457841326", "TR330006100519786457841327",
             "TR000006100519786457841326", "TR990006100519786457841326", "DE330006100519786457841326",
             "TR33 00061005 1978 6457 8413 26", "TR33 0006 1005 1978 6457 8413 2A", "TR3A 0006 1005 1978 6457 8413 26",
             "TR33 00A6 1005 1978 6457 8413 26", "TR33 0006 1005 1978 6457 8413 26 00", "%3'den", "%3,25'den",
             "%3'ten'de", "25 TL'dan", "5 kg'den", "1.'nın", "4.'ya", "1,005 USD", "25 JPY", "25 XAU", "25 USD EUR",
             "$25 USD", "25€ EUR", "5 kg + 2 kg", "10-15 kişi = 2", "5 MG", "Prof.'e'nin'den", "vb.'nin'in", "5 Μg",
             "5 km/s", "5 m^2", "2 sa'tan", "2 sa'ta", "1e3 kg", "90 mph", "3 + 4 = 7", "0850 22 33 44", "tel 123",
             "0532 123 45 6X", "+90 1 2", "www.örnek.com", "https://xn--rnek-4qa.com/a", "info@xn--rnek-4qa.com",
             "user@örnek.com", "\"user\"@ornek.com", "\"user name\"@ornek.com", "ftp://ornek.com/a",
             "https://user:secret@ornek.com/a", "https://ornek.com/%ZZ", "https://ornek.com:99999/a",
             "https://ornek.invalid/a", "ornek.com",
         })
{
    Expect(input, input, complete: false);
    var issues = Run(input, AmbiguityPolicy.Preserve)!.Issues;
    try
    {
        Run(input, AmbiguityPolicy.Reject);
        failures.Add($"reject \"{input}\": no error");
    }
    catch (NormalizeException e)
    {
        Check(e.Kind == NormalizeErrorKind.Unresolved && e.Issues.SequenceEqual(issues), $"reject \"{input}\"");
    }
}
foreach (var (input, kind) in new[]
         {
             ("IIII", HintKind.Roman), ("VX", HintKind.Roman), ("MMMM", HintKind.Roman), ("25", HintKind.Ordinal),
             ("1,2.", HintKind.Ordinal), ("12:34", HintKind.Range), ("+44 5321234567", HintKind.Telephone),
             ("https://ornek.com/%G0", HintKind.Electronic),
         })
{
    ExpectError(input, "InvalidHint", hint: kind);
}

// 4) Rastgele Unicode: fallback'te ya tam sonuç ya InvalidInput (tests/fallback.rs proptest değişmezleri)
{
    var random = new Random(20261007);
    for (var n = 0; n < 512; n++)
    {
        var sb = new StringBuilder();
        var count = random.Next(0, 100);
        for (var i = 0; i < count; i++)
        {
            int cp;
            do
            {
                cp = random.Next(4) switch
                {
                    0 => random.Next(0x20, 0x7F),
                    1 => random.Next(0xA0, 0x2000),
                    2 => random.Next(0x2000, 0x10000),
                    _ => random.Next(0x10000, 0x110000),
                };
            } while (cp is >= 0xD800 and <= 0xDFFF);
            sb.Append(char.ConvertFromUtf32(cp));
        }
        var input = sb.ToString();
        try
        {
            var result = Run(input, F)!;
            var end = 0;
            var ok = result.Complete && result.Issues.Count == 0;
            foreach (var segment in result.Segments)
            {
                ok &= segment.Range.Start == end && segment.Text.Length > 0 && segment.Kind != SegmentKind.Unresolved;
                end = segment.Range.End;
            }
            ok &= end == Encoding.UTF8.GetByteCount(input);
            ok &= string.Concat(result.Segments.Select(s => s.Text)) == result.NormalizedText;
            ok &= result.Segments.Where(s => s.Kind == SegmentKind.Fallback).Select(s => s.Range)
                .SequenceEqual(result.Fallbacks.Select(f => f.Range));
            Check(ok, $"random invariants #{n}: {Escape(input)}");
        }
        catch (NormalizeException e)
        {
            Check(e.Kind == NormalizeErrorKind.InvalidInput, $"random #{n} {e.DebugName}: {Escape(input)}");
        }
        catch (Exception e)
        {
            failures.Add($"random #{n} {e.GetType().Name}: {e.Message}: {Escape(input)}");
        }
    }
}

Console.WriteLine($"passed: {passed}, failed: {failures.Count}");
foreach (var failure in failures)
{
    Console.WriteLine("FAIL " + failure);
}
return failures.Count == 0 ? 0 : 1;

static JsonObject RangeJson(SourceRange range) => new() { ["end"] = range.End, ["start"] = range.Start };

static JsonArray IssuesJson(IReadOnlyList<Issue> issues) => new(issues.Select(i => (JsonNode)new JsonObject
{
    ["category"] = i.Category.ToString(),
    ["explanation"] = i.Explanation,
    ["range"] = RangeJson(i.Range),
}).ToArray());

static bool Same(JsonNode? a, JsonNode? b)
{
    switch (a)
    {
        case null:
            return b == null;
        case JsonObject oa when b is JsonObject ob:
            return oa.Count == ob.Count && oa.All(p => ob.ContainsKey(p.Key) && Same(p.Value, ob[p.Key]));
        case JsonArray aa when b is JsonArray ab:
            return aa.Count == ab.Count && aa.Zip(ab).All(p => Same(p.First, p.Second));
        case JsonValue va when b is JsonValue vb:
            return va.ToJsonString() == vb.ToJsonString();
        default:
            return false;
    }
}

static string Escape(string s) =>
    string.Concat(s.EnumerateRunes().Select(r => r.Value < 0x80 && r.Value >= 0x20 ? r.ToString() : $"\\u{{{r.Value:X}}}"));

static string FindRepo()
{
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir != null)
    {
        var candidate = Path.Combine(dir.FullName, "github", "normalizer-tr");
        if (Directory.Exists(candidate))
        {
            return candidate;
        }
        dir = dir.Parent;
    }
    throw new DirectoryNotFoundException("github\\normalizer-tr bulunamadı; yolu argüman olarak verin.");
}
