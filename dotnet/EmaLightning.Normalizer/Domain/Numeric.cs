// src/domain/numeric.rs portu
using System.Globalization;
using System.Text;

namespace NormalizerTr;

internal abstract record NumericPreference;

internal sealed record NumberPreference(Numeric Number) : NumericPreference;

internal sealed record SentenceNumberPreference(Number Number) : NumericPreference;

internal sealed record NumericFailure(IssueCategory Category, NumericPreference? Preference = null);

internal sealed class Numeric
{
    readonly Number number;
    readonly Inflection? @case;

    Numeric(Number number, bool ordinal, Inflection? @case)
    {
        this.number = number;
        Ordinal = ordinal;
        this.@case = @case;
    }

    public bool Ordinal { get; }

    public static Res<Numeric, NumericFailure> Automatic(string text)
    {
        if (Uni.Any(text, r => Uni.IsNumeric(r) && !Uni.IsAsciiDigit(r)))
        {
            return Res<Numeric, NumericFailure>.Err(new(IssueCategory.Unsupported));
        }
        if (Parse(text, false) is { } value)
        {
            return Res<Numeric, NumericFailure>.Ok(value);
        }
        if (!NumericText.SplitSuffix(text, out var @base, out var suffix))
        {
            return Res<Numeric, NumericFailure>.Err(new(IssueCategory.InvalidExpression));
        }
        if (@base.EndsWith('.') && suffix == null)
        {
            var sentence = Number.Parse(@base[..^1]);
            return Res<Numeric, NumericFailure>.Err(sentence != null && sentence.Fraction.Length == 0 && !sentence.Grouped
                ? new NumericFailure(IssueCategory.Ambiguous, new SentenceNumberPreference(sentence))
                : new NumericFailure(IssueCategory.InvalidExpression));
        }
        var number = Number.Parse(@base);
        if (number == null)
        {
            return Res<Numeric, NumericFailure>.Err(new(IssueCategory.InvalidExpression));
        }
        if (number.Grouped)
        {
            // Rust: Option<Option<Inflection>> — dış null tercih yok demek
            bool found;
            Inflection? inner = null;
            if (suffix == null)
            {
                found = true;
            }
            else if (number.Fraction.Length == 0
                     && Morphology.IntegerInflection(Numerals.NumberSpoken(number), suffix) is { } family)
            {
                found = true;
                inner = family;
            }
            else
            {
                found = false;
            }
            return Res<Numeric, NumericFailure>.Err(new NumericFailure(
                IssueCategory.Ambiguous,
                found
                    ? new NumberPreference(new Numeric(number, inner == Inflection.Ordinal,
                        inner == Inflection.Ordinal ? null : inner))
                    : null));
        }
        Inflection? @case = null;
        if (suffix != null)
        {
            if (number.Fraction.Length > 0)
            {
                return Res<Numeric, NumericFailure>.Err(new(IssueCategory.Unsupported));
            }
            @case = Morphology.IntegerInflection(Numerals.NumberSpoken(number), suffix);
            if (@case == null)
            {
                return Res<Numeric, NumericFailure>.Err(new(IssueCategory.InvalidExpression));
            }
        }
        return Res<Numeric, NumericFailure>.Ok(new Numeric(number, false, @case));
    }

    public static Numeric? CardinalHint(string text)
    {
        if (!NumericText.SplitSuffix(text, out var @base, out var suffix))
        {
            return null;
        }
        var number = Number.ParseCardinalHint(@base);
        if (number == null || number.Fraction.Length > 0)
        {
            return null;
        }
        Inflection? family = null;
        if (suffix != null)
        {
            family = Morphology.IntegerInflection(Numerals.NumberSpoken(number), suffix);
            if (family == null)
            {
                return null;
            }
        }
        return new Numeric(number, family == Inflection.Ordinal, family == Inflection.Ordinal ? null : family);
    }

    public static Numeric? Parse(string text, bool ordinalHint)
    {
        if (!NumericText.SuffixParts(text, out var @base, out var suffixes))
        {
            return null;
        }
        var period = @base.EndsWith('.');
        if (ordinalHint && !period && suffixes.Count == 0)
        {
            return null;
        }
        var body = period ? @base[..^1] : @base;
        var number = Number.Parse(body);
        if (number == null || number.Fraction.Length > 0 || number.Grouped)
        {
            return null;
        }
        var spoken = Numerals.NumberSpoken(number);
        bool ordinal;
        Inflection? @case;
        if (period)
        {
            if (suffixes.Count == 0 && !ordinalHint)
            {
                return null;
            }
            if (suffixes.Count > 1)
            {
                return null;
            }
            spoken.Inflect(Inflection.Ordinal);
            ordinal = true;
            @case = null;
            if (suffixes.Count == 1)
            {
                @case = Morphology.SpokenCase(spoken, suffixes[0]);
                if (@case == null)
                {
                    return null;
                }
            }
        }
        else if (suffixes.Count > 0)
        {
            var first = suffixes[0];
            if (Morphology.IntegerInflection(spoken, first) is not { } firstFamily)
            {
                if (suffixes.Count != 1)
                {
                    return null;
                }
                var ordinalSuffix = spoken.SourceSuffix(Inflection.Ordinal);
                if (!first.StartsOrd(ordinalSuffix))
                {
                    return null;
                }
                var remaining = first[ordinalSuffix.Length..];
                spoken.Inflect(Inflection.Ordinal);
                var remainingCase = Morphology.SpokenCase(spoken, remaining);
                return remainingCase == null ? null : new Numeric(number, true, remainingCase);
            }
            ordinal = firstFamily == Inflection.Ordinal;
            if (ordinal)
            {
                spoken.Inflect(Inflection.Ordinal);
                @case = null;
                if (suffixes.Count > 1)
                {
                    @case = Morphology.SpokenCase(spoken, suffixes[1]);
                    if (@case == null)
                    {
                        return null;
                    }
                }
            }
            else
            {
                if (ordinalHint || suffixes.Count != 1)
                {
                    return null;
                }
                @case = firstFamily;
            }
        }
        else
        {
            ordinal = ordinalHint;
            @case = null;
        }
        return new Numeric(number, ordinal, @case);
    }

    public static Numeric? Roman(string text)
    {
        if (!NumericText.SuffixParts(text, out var @base, out var suffixes))
        {
            return null;
        }
        var ordinal = @base.EndsWith('.');
        var roman = ordinal ? @base[..^1] : @base;
        if (NumericText.RomanValue(roman) is not { } value)
        {
            return null;
        }
        var numeric = new StringBuilder(value.ToString(CultureInfo.InvariantCulture));
        if (ordinal)
        {
            numeric.Append('.');
        }
        foreach (var suffix in suffixes)
        {
            numeric.Append('\'').Append(suffix);
        }
        var parsed = Parse(numeric.ToString(), ordinal);
        if (parsed == null || (suffixes.Count > 0 && !parsed.Ordinal))
        {
            return null;
        }
        return parsed;
    }

    public Spoken Render()
    {
        var spoken = Numerals.NumberSpoken(number);
        if (Ordinal)
        {
            spoken.Inflect(Inflection.Ordinal);
        }
        if (@case is { } c)
        {
            spoken.Inflect(c);
        }
        return spoken;
    }

    public SegmentKind Kind =>
        Ordinal ? SegmentKind.Ordinal : number.Fraction.Length == 0 ? SegmentKind.Cardinal : SegmentKind.Decimal;
}

internal sealed class Quantity
{
    readonly Number? plain;
    readonly Amount? amount;
    readonly Currency currency;
    readonly Lexeme lexeme;
    readonly Lexeme? prefix;
    readonly Inflection? @case;

    Quantity(Number? plain, Amount? amount, Currency currency, Lexeme lexeme, Lexeme? prefix, Inflection? @case)
    {
        this.plain = plain;
        this.amount = amount;
        this.currency = currency;
        this.lexeme = lexeme;
        this.prefix = prefix;
        this.@case = @case;
    }

    public static Quantity? Parse(string number, string label)
    {
        if (!NumericText.SuffixParts(label, out var @base, out var suffixes) || suffixes.Count > 1)
        {
            return null;
        }
        Number? plain = null;
        Amount? amount = null;
        var currency = Currency.Try;
        Lexeme lexeme;
        Lexeme? prefix = null;
        if (Lexicon.ParseCurrency(@base) is { } parsed)
        {
            amount = Amount.Parse(number);
            if (amount == null)
            {
                return null;
            }
            currency = parsed;
            lexeme = Lexicon.CurrencyLexeme(parsed, @base);
        }
        else if (Lexicon.Rate(@base) is { } rate)
        {
            if (suffixes.Count > 0)
            {
                return null;
            }
            plain = Number.Parse(number);
            if (plain == null)
            {
                return null;
            }
            lexeme = rate.Unit;
            prefix = rate.Prefix;
        }
        else
        {
            plain = Number.Parse(number);
            if (plain == null || Lexicon.Unit(@base) is not { } unit)
            {
                return null;
            }
            lexeme = unit;
        }
        Inflection? @case = null;
        if (suffixes.Count == 1)
        {
            @case = Morphology.CaseInflection(lexeme.Source, suffixes[0]);
            if (@case == null)
            {
                return null;
            }
        }
        return new Quantity(plain, amount, currency, lexeme, prefix, @case);
    }

    public Spoken Render()
    {
        Spoken spoken;
        if (amount == null)
        {
            var text = $"{Numerals.NumberSpoken(plain!).IntoText()} {lexeme.Output}";
            spoken = Spoken.Lexical(text, lexeme.Target);
        }
        else
        {
            spoken = Numerals.AmountSpoken(amount, lexeme.Output, lexeme.Target,
                Lexicon.CurrencyMinor(currency).Target);
        }
        if (prefix is { } p)
        {
            var denominator = Spoken.Lexical(p.Output, p.Target);
            denominator.Inflect(Inflection.Locative);
            spoken.Prefix($"{denominator.IntoText()} ");
        }
        if (@case is { } c)
        {
            spoken.Inflect(c);
        }
        return spoken;
    }

    public bool IsMoney => amount != null;
}

internal sealed class NumericRange
{
    readonly Number start;
    readonly Number end;
    readonly Lexeme? context;
    readonly string? noun;

    NumericRange(Number start, Number end, Lexeme? context, string? noun)
    {
        this.start = start;
        this.end = end;
        this.context = context;
        this.noun = noun;
    }

    public static NumericRange? Parse(string text, string? context)
    {
        (Number Start, Number End)? found = null;
        for (var offset = 0; offset < text.Length; offset++)
        {
            var ch = text[offset];
            if (ch is not ('-' or '–') || offset == 0)
            {
                continue;
            }
            var start = Number.Parse(text[..offset]);
            var end = Number.Parse(text[(offset + 1)..]);
            if (start != null && end != null)
            {
                if (found != null)
                {
                    return null;
                }
                found = (start, end);
            }
        }
        if (found is not { } pair)
        {
            return null;
        }
        Lexeme? unit = null;
        string? noun = null;
        if (context != null)
        {
            if (NumericText.IsCountNoun(context))
            {
                noun = context;
            }
            else
            {
                unit = Lexicon.Unit(context);
                if (unit == null)
                {
                    return null;
                }
            }
        }
        return new NumericRange(pair.Start, pair.End, unit, noun);
    }

    public string Render()
    {
        var result = new StringBuilder(
            $"{Numerals.NumberSpoken(start).IntoText()} ila {Numerals.NumberSpoken(end).IntoText()}");
        if (context is { } unit)
        {
            result.Append(' ').Append(unit.Output);
        }
        if (noun != null)
        {
            result.Append(' ').Append(noun);
        }
        return result.ToString();
    }
}

/// <summary>numeric.rs serbest fonksiyonları.</summary>
internal static class NumericText
{
    static readonly string[] CountNouns = { "kişi", "adet", "gün", "yaş" };

    public static bool IsCountNoun(string text) => CountNouns.Contains(Lexicon.LookupKey(text));

    public static bool SuffixParts(string text, out string @base, out List<string> suffixes)
    {
        var parts = text.Split(Uni.Apostrophes);
        @base = parts[0];
        suffixes = parts.Skip(1).ToList();
        return suffixes.Count <= 2 && suffixes.All(s => s.Length > 0);
    }

    public static bool SplitSuffix(string text, out string @base, out string? suffix)
    {
        var parts = text.Split(Uni.Apostrophes);
        @base = parts[0];
        suffix = parts.Length > 1 ? parts[1] : null;
        return parts.Length <= 2 && suffix is not "";
    }

    public static string? DigitsHint(string text)
    {
        var body = text.StartsWith('+') ? text[1..] : text;
        return Uni.AnyAsciiDigit(body)
               && Uni.AllChars(body, c => Uni.IsAsciiDigit(c) || c is ' ' or '.' or '/' or '-' or '(' or ')')
            ? text
            : null;
    }

    public static int? RomanValue(string text)
    {
        if (text.Length == 0 || Uni.Len8(text) > 15)
        {
            return null;
        }
        var total = 0;
        var previous = 0;
        for (var i = text.Length - 1; i >= 0; i--)
        {
            int n = text[i] switch
            {
                'I' => 1,
                'V' => 5,
                'X' => 10,
                'L' => 50,
                'C' => 100,
                'D' => 500,
                'M' => 1000,
                _ => 0,
            };
            if (n == 0)
            {
                return null;
            }
            total += n < previous ? -n : n;
            previous = n;
        }
        if (total is < 1 or > 3999)
        {
            return null;
        }
        var rest = total;
        var canonical = new StringBuilder();
        foreach (var (n, s) in new[]
                 {
                     (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
                     (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
                 })
        {
            while (rest >= n)
            {
                canonical.Append(s);
                rest -= n;
            }
        }
        return canonical.ToString() == text ? total : null;
    }

    public static bool Label(string text)
    {
        var @base = Uni.Head(text);
        return Lexicon.ParseCurrency(@base) != null || Lexicon.Unit(@base) != null || Lexicon.Rate(@base) != null;
    }

    public static bool UnsupportedLabel(string text) => text is "JPY" or "CHF" or "CAD" or "AUD" or "RUB" or "¥"
        or "Hz" or "kHz" or "MHz" or "GHz" or "°C" or "°F" or "mph" or "GB" or "MB" or "V" or "A" or "W";

    /// <summary>null = Rust None; aksi halde Ok/Err.</summary>
    public static Res<(Lexeme Entry, Inflection? Case), IssueCategory>? LexicalReading(string text)
    {
        var @base = Uni.Head(text);
        Lexeme entry;
        if (Lexicon.Abbreviation(@base) is { } abbreviation)
        {
            entry = abbreviation;
        }
        else if (Lexicon.ParseCurrency(@base) is { } currency)
        {
            entry = Lexicon.CurrencyLexeme(currency, @base);
        }
        else
        {
            return null;
        }
        if (!SuffixParts(text, out _, out var suffixes) || suffixes.Count > 1)
        {
            return Res<(Lexeme, Inflection?), IssueCategory>.Err(IssueCategory.Unsupported);
        }
        Inflection? @case = null;
        if (suffixes.Count == 1)
        {
            @case = Morphology.CaseInflection(entry.Source, suffixes[0]);
            if (@case == null)
            {
                return Res<(Lexeme, Inflection?), IssueCategory>.Err(IssueCategory.InvalidExpression);
            }
        }
        return Res<(Lexeme, Inflection?), IssueCategory>.Ok((entry, @case));
    }

    public static Res<(Number Number, Inflection? Case), IssueCategory>? Percent(string text)
    {
        if (!SuffixParts(text, out var @base, out var suffixes) || !@base.StartsWith('%'))
        {
            return null;
        }
        var number = Number.Parse(@base[1..]);
        if (number == null)
        {
            return Res<(Number, Inflection?), IssueCategory>.Err(IssueCategory.InvalidExpression);
        }
        if (suffixes.Count > 1)
        {
            return Res<(Number, Inflection?), IssueCategory>.Err(IssueCategory.Unsupported);
        }
        var spoken = Numerals.NumberSpoken(number);
        Inflection? inflection = null;
        if (suffixes.Count == 1)
        {
            var s = suffixes[0];
            if (s == spoken.SourceSuffix(Inflection.Derivation))
            {
                inflection = Inflection.Derivation;
            }
            else
            {
                inflection = Morphology.SpokenCase(spoken, s);
                if (inflection == null)
                {
                    return Res<(Number, Inflection?), IssueCategory>.Err(IssueCategory.InvalidExpression);
                }
            }
        }
        return Res<(Number, Inflection?), IssueCategory>.Ok((number, inflection));
    }
}
