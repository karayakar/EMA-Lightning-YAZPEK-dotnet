// src/domain/lexicon.rs portu
using System.Text;

namespace NormalizerTr;

internal readonly record struct Lexeme(string Output, Word Source, Word Target)
{
    public static Lexeme Same(string output, Harmony harmony, WordEnd end)
    {
        var word = new Word(output, harmony, end);
        return new Lexeme(output, word, word);
    }

    public static Lexeme Distinct(string output, Word source, Word target) => new(output, source, target);
}

internal enum Currency
{
    Try,
    Usd,
    Eur,
    Gbp,
}

internal static class Lexicon
{
    public static readonly string[] Months =
    {
        "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
    };

    public static string? MonthName(int month) => month is >= 1 and <= 12 ? Months[month - 1] : null;

    /// <summary>Turkish casing is confined to explicit contextual lookup keys, never source rewriting.</summary>
    public static string LookupKey(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            if (rune.Value == 'I')
            {
                sb.Append('ı');
            }
            else if (rune.Value == 'İ')
            {
                sb.Append('i');
            }
            else
            {
                sb.Append(Rune.ToLowerInvariant(rune).ToString());
            }
        }
        return sb.ToString();
    }

    static readonly (string Key, Lexeme Value)[] Units =
    {
        ("kg", Lexeme.Same("kilogram", Harmony.BackFlat, WordEnd.Voiced)),
        ("g", Lexeme.Same("gram", Harmony.BackFlat, WordEnd.Voiced)),
        ("gr", Lexeme.Same("gram", Harmony.BackFlat, WordEnd.Voiced)),
        ("mg", Lexeme.Same("miligram", Harmony.BackFlat, WordEnd.Voiced)),
        ("µg", Lexeme.Same("mikrogram", Harmony.BackFlat, WordEnd.Voiced)),
        ("μg", Lexeme.Same("mikrogram", Harmony.BackFlat, WordEnd.Voiced)),
        ("km", Lexeme.Same("kilometre", Harmony.FrontFlat, WordEnd.Vowel)),
        ("m", Lexeme.Same("metre", Harmony.FrontFlat, WordEnd.Vowel)),
        ("cm", Lexeme.Same("santimetre", Harmony.FrontFlat, WordEnd.Vowel)),
        ("mm", Lexeme.Same("milimetre", Harmony.FrontFlat, WordEnd.Vowel)),
        ("L", Lexeme.Same("litre", Harmony.FrontFlat, WordEnd.Vowel)),
        ("lt", Lexeme.Same("litre", Harmony.FrontFlat, WordEnd.Vowel)),
        ("mL", Lexeme.Same("mililitre", Harmony.FrontFlat, WordEnd.Vowel)),
        ("ml", Lexeme.Same("mililitre", Harmony.FrontFlat, WordEnd.Vowel)),
        ("dk", Lexeme.Same("dakika", Harmony.BackFlat, WordEnd.Vowel)),
        ("sn", Lexeme.Same("saniye", Harmony.FrontFlat, WordEnd.Vowel)),
        ("sa", Lexeme.Same("saat", Harmony.FrontFlat, WordEnd.Voiceless)),
        ("m²", Lexeme.Same("metrekare", Harmony.FrontFlat, WordEnd.Vowel)),
        ("cm²", Lexeme.Same("santimetrekare", Harmony.FrontFlat, WordEnd.Vowel)),
        ("km²", Lexeme.Same("kilometrekare", Harmony.FrontFlat, WordEnd.Vowel)),
        ("m³", Lexeme.Same("metreküp", Harmony.FrontRound, WordEnd.SoftensP)),
    };

    public static Lexeme? Unit(string symbol)
    {
        foreach (var (key, value) in Units)
        {
            if (key == symbol)
            {
                return value;
            }
        }
        return null;
    }

    /// <summary>Detection of an unsupported spelling is not acceptance or source case rewriting.</summary>
    public static bool UnitMarker(string symbol)
    {
        foreach (var (key, _) in Units)
        {
            if (Uni.EqIgnoreAsciiCase(key, symbol))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>(payda, birim)</summary>
    public static (Lexeme Prefix, Lexeme Unit)? Rate(string symbol) => symbol switch
    {
        "km/sa" or "km/h" => (Unit("sa")!.Value, Unit("km")!.Value),
        "m/s" => (Unit("sn")!.Value, Unit("m")!.Value),
        _ => null,
    };

    public static Lexeme? Abbreviation(string symbol) => symbol switch
    {
        "Dr." => Lexeme.Same("doktor", Harmony.BackRound, WordEnd.Voiced),
        "Prof." => Lexeme.Same("profesör", Harmony.FrontRound, WordEnd.Voiced),
        "vb." => Lexeme.Distinct("ve benzeri", new Word("be", Harmony.FrontFlat, WordEnd.Vowel),
            new Word("benzeri", Harmony.FrontFlat, WordEnd.Vowel)),
        "TBMM" => Lexeme.Distinct("te be me me", new Word("me", Harmony.FrontFlat, WordEnd.Vowel),
            new Word("me", Harmony.FrontFlat, WordEnd.Vowel)),
        "PTT" => Lexeme.Distinct("pe te te", new Word("te", Harmony.FrontFlat, WordEnd.Vowel),
            new Word("te", Harmony.FrontFlat, WordEnd.Vowel)),
        "NATO" => Lexeme.Same("nato", Harmony.BackRound, WordEnd.Vowel),
        "IBAN" => Lexeme.Same("iban", Harmony.BackFlat, WordEnd.Voiced),
        "KDV" => Lexeme.Distinct("katma değer vergisi", new Word("ve", Harmony.FrontFlat, WordEnd.Vowel),
            new Word("vergisi", Harmony.FrontFlat, WordEnd.Possessive)),
        _ => null,
    };

    public static Currency? ParseCurrency(string label) => label switch
    {
        "TL" or "TRY" or "₺" => Currency.Try,
        "USD" or "$" => Currency.Usd,
        "EUR" or "€" => Currency.Eur,
        "GBP" or "£" => Currency.Gbp,
        _ => null,
    };

    public static Lexeme CurrencyLexeme(Currency currency, string label) => currency switch
    {
        Currency.Try => Lexeme.Distinct(
            "Türk lirası",
            label == "TRY" ? new Word("ye", Harmony.FrontFlat, WordEnd.Vowel)
            : label == "₺" ? new Word("lira", Harmony.BackFlat, WordEnd.Vowel)
            : new Word("le", Harmony.FrontFlat, WordEnd.Vowel),
            new Word("lirası", Harmony.BackFlat, WordEnd.Possessive)),
        Currency.Usd => Lexeme.Distinct(
            "dolar",
            label == "USD" ? new Word("de", Harmony.FrontFlat, WordEnd.Vowel)
            : new Word("dolar", Harmony.BackFlat, WordEnd.Voiced),
            new Word("dolar", Harmony.BackFlat, WordEnd.Voiced)),
        Currency.Eur => Lexeme.Distinct(
            "avro",
            label == "EUR" ? new Word("re", Harmony.FrontFlat, WordEnd.Vowel)
            : new Word("avro", Harmony.BackRound, WordEnd.Vowel),
            new Word("avro", Harmony.BackRound, WordEnd.Vowel)),
        _ => Lexeme.Distinct(
            "sterlin",
            label == "GBP" ? new Word("pe", Harmony.FrontFlat, WordEnd.Vowel)
            : new Word("sterlin", Harmony.FrontFlat, WordEnd.Voiced),
            new Word("sterlin", Harmony.FrontFlat, WordEnd.Voiced)),
    };

    public static Lexeme CurrencyMinor(Currency currency) => currency switch
    {
        Currency.Try => Lexeme.Same("kuruş", Harmony.BackRound, WordEnd.Voiceless),
        Currency.Usd or Currency.Eur => Lexeme.Same("sent", Harmony.FrontFlat, WordEnd.Voiceless),
        _ => Lexeme.Same("peni", Harmony.FrontFlat, WordEnd.Vowel),
    };
}
