// src/numerals.rs portu
namespace NormalizerTr;

internal enum Sign
{
    None,
    Plus,
    Minus,
}

internal sealed class Amount
{
    Amount(Sign sign, ulong major, int minor)
    {
        Sign = sign;
        Major = major;
        Minor = minor;
    }

    public Sign Sign { get; }
    public ulong Major { get; }
    public int Minor { get; }

    public static Amount? Parse(string text)
    {
        var number = Number.Parse(text);
        if (number?.Minor() is not { } minor)
        {
            return null;
        }
        return new Amount(number.Sign, number.Integer, minor);
    }
}

internal sealed class Number
{
    public const ulong MagnitudeLimit = 1_000_000_000_000_000_000;

    Number(Sign sign, ulong integer, string fraction, bool grouped)
    {
        Sign = sign;
        Integer = integer;
        Fraction = fraction;
        Grouped = grouped;
    }

    public Sign Sign { get; }
    public ulong Integer { get; }
    public string Fraction { get; }
    public bool Grouped { get; }

    public static Number? Parse(string text) => ParseWithPadding(text, false);

    public static Number? ParseCardinalHint(string text) => ParseWithPadding(text, true);

    static Number? ParseWithPadding(string text, bool allowPadding)
    {
        var sign = Sign.None;
        var body = text;
        if (text.Length > 0 && text[0] == '-')
        {
            sign = Sign.Minus;
            body = text[1..];
        }
        else if (text.Length > 0 && text[0] == '+')
        {
            sign = Sign.Plus;
            body = text[1..];
        }
        var parts = body.Split(',');
        if (parts.Length > 2)
        {
            return null;
        }
        var whole = parts[0];
        var fraction = parts.Length > 1 ? parts[1] : null;
        if (fraction != null && (Uni.Len8(fraction) is < 1 or > 9 || !Uni.AllAsciiDigits(fraction)))
        {
            return null;
        }
        var grouped = whole.Contains('.');
        ulong integer = 0;
        var groups = whole.Split('.');
        for (var index = 0; index < groups.Length; index++)
        {
            var group = groups[index];
            if (group.Length == 0
                || !Uni.AllAsciiDigits(group)
                || (!allowPadding && index == 0 && group.Length > 1 && group[0] == '0')
                || (grouped && index == 0 && group.Length > 3)
                || (index > 0 && group.Length != 3))
            {
                return null;
            }
            foreach (var digit in group)
            {
                integer = integer * 10 + (ulong)(digit - '0');
                if (integer >= MagnitudeLimit)
                {
                    return null;
                }
            }
        }
        return new Number(sign, integer, fraction ?? "", grouped);
    }

    public int? Minor() => Fraction.Length switch
    {
        0 => 0,
        1 => (Fraction[0] - '0') * 10,
        2 => (Fraction[0] - '0') * 10 + (Fraction[1] - '0'),
        _ => null,
    };
}

internal static class Numerals
{
    static readonly Word[] Digits =
    {
        new("sıfır", Harmony.BackFlat, WordEnd.Voiced),
        new("bir", Harmony.FrontFlat, WordEnd.Voiced),
        new("iki", Harmony.FrontFlat, WordEnd.Vowel),
        new("üç", Harmony.FrontRound, WordEnd.Voiceless),
        new("dört", Harmony.FrontRound, WordEnd.Softens),
        new("beş", Harmony.FrontFlat, WordEnd.Voiceless),
        new("altı", Harmony.BackFlat, WordEnd.Vowel),
        new("yedi", Harmony.FrontFlat, WordEnd.Vowel),
        new("sekiz", Harmony.FrontFlat, WordEnd.Voiced),
        new("dokuz", Harmony.BackRound, WordEnd.Voiced),
    };

    static readonly Word[] Tens =
    {
        new("on", Harmony.BackRound, WordEnd.Voiced),
        new("yirmi", Harmony.FrontFlat, WordEnd.Vowel),
        new("otuz", Harmony.BackRound, WordEnd.Voiced),
        new("kırk", Harmony.BackFlat, WordEnd.Voiceless),
        new("elli", Harmony.FrontFlat, WordEnd.Vowel),
        new("altmış", Harmony.BackFlat, WordEnd.Voiceless),
        new("yetmiş", Harmony.FrontFlat, WordEnd.Voiceless),
        new("seksen", Harmony.FrontFlat, WordEnd.Voiced),
        new("doksan", Harmony.BackFlat, WordEnd.Voiced),
    };

    static readonly Word Hundred = new("yüz", Harmony.FrontRound, WordEnd.Voiced);

    static readonly Word[] Scales =
    {
        new("bin", Harmony.FrontFlat, WordEnd.Voiced),
        new("milyon", Harmony.BackRound, WordEnd.Voiced),
        new("milyar", Harmony.BackFlat, WordEnd.Voiced),
        new("trilyon", Harmony.BackRound, WordEnd.Voiced),
        new("katrilyon", Harmony.BackRound, WordEnd.Voiced),
    };

    static void PushGroup(int value, List<Word> words)
    {
        if (value >= 100)
        {
            if (value / 100 > 1)
            {
                words.Add(Digits[value / 100]);
            }
            words.Add(Hundred);
        }
        var rest = value % 100;
        if (rest >= 10)
        {
            words.Add(Tens[rest / 10 - 1]);
        }
        if (rest % 10 != 0)
        {
            words.Add(Digits[rest % 10]);
        }
    }

    /// <summary>Only called with validated domain magnitudes or bounded calendar/clock fields.</summary>
    public static Spoken Cardinal(ulong value)
    {
        if (value == 0)
        {
            return Spoken.FromWords(new[] { Digits[0] });
        }
        var groups = new int[6];
        var rest = value;
        for (var i = 0; i < groups.Length; i++)
        {
            groups[i] = (int)(rest % 1000);
            rest /= 1000;
        }
        var words = new List<Word>(24);
        for (var index = groups.Length - 1; index >= 0; index--)
        {
            var group = groups[index];
            if (group == 0)
            {
                continue;
            }
            if (index != 1 || group != 1)
            {
                PushGroup(group, words);
            }
            if (index > 0)
            {
                words.Add(Scales[index - 1]);
            }
        }
        return Spoken.FromWords(words);
    }

    public static string SignText(Sign sign) => sign switch
    {
        Sign.Plus => "artı ",
        Sign.Minus => "eksi ",
        _ => "",
    };

    public static Spoken NumberSpoken(Number number)
    {
        var spoken = Cardinal(number.Integer);
        spoken.Prefix(SignText(number.Sign));
        if (number.Fraction.Length > 0)
        {
            spoken.AppendLiteral(" virgül");
            foreach (var digit in number.Fraction)
            {
                spoken.AppendWord(Digits[digit - '0']);
            }
        }
        return spoken;
    }

    public static Spoken AmountSpoken(Amount amount, string major, Word majorTail, Word minorTail)
    {
        var text = $"{SignText(amount.Sign)}{Cardinal(amount.Major).IntoText()} {major}";
        var spoken = Spoken.Lexical(text, majorTail);
        if (amount.Minor != 0)
        {
            spoken.AppendLiteral($" {Cardinal((ulong)amount.Minor).IntoText()}");
            spoken.AppendWord(minorTail);
        }
        return spoken;
    }

    public static string DigitsText(string text)
    {
        var result = new System.Text.StringBuilder();
        foreach (var ch in text)
        {
            string word;
            if (ch == '+')
            {
                word = "artı";
            }
            else if (Uni.IsAsciiDigit(ch))
            {
                word = Digits[ch - '0'].Text;
            }
            else
            {
                continue;
            }
            if (result.Length > 0)
            {
                result.Append(' ');
            }
            result.Append(word);
        }
        return result.ToString();
    }

    public static string? DigitName(char digit) => Uni.IsAsciiDigit(digit) ? Digits[digit - '0'].Text : null;
}
