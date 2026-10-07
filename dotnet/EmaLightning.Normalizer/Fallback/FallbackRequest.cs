// src/fallback/mod.rs portu
using System.Text;

namespace NormalizerTr;

internal enum LetterReading
{
    Spell,
    PreserveWords,
}

/// <summary>Keeps the failed family and readable source alternative together.</summary>
internal sealed class FallbackRequest
{
    // Rust Prepared varyantları: yalnızca biri dolu
    readonly NumericPreference? number;
    readonly TemporalPreference? temporal;
    readonly DateSurface? date;
    readonly TimeSurface? time;
    readonly LetterReading literal;

    FallbackRequest(IssueCategory? category, FallbackClass @class, FallbackReason reason,
        NumericPreference? number = null, TemporalPreference? temporal = null, DateSurface? date = null,
        TimeSurface? time = null, LetterReading literal = LetterReading.PreserveWords)
    {
        Category = category;
        Class = @class;
        Reason = reason;
        this.number = number;
        this.temporal = temporal;
        this.date = date;
        this.time = time;
        this.literal = literal;
    }

    public IssueCategory? Category { get; }
    public FallbackClass Class { get; }
    public FallbackReason Reason { get; }

    public static FallbackRequest Unresolved(string source, FallbackClass @class, IssueCategory category)
    {
        var date = @class == FallbackClass.Date ? DateSurface.Parse(source) : null;
        var time = @class == FallbackClass.Time ? TimeSurface.Parse(source) : null;
        var spell = @class is FallbackClass.Identifier or FallbackClass.Abbreviation or FallbackClass.Roman
            or FallbackClass.Electronic;
        return Primary(source, @class, category, date: date, time: time,
            literal: spell ? LetterReading.Spell : LetterReading.PreserveWords);
    }

    static FallbackRequest Primary(string source, FallbackClass @class, IssueCategory category,
        NumericPreference? number = null, TemporalPreference? temporal = null, DateSurface? date = null,
        TimeSurface? time = null, LetterReading literal = LetterReading.PreserveWords)
    {
        var reason = Uni.Len8(source) > 1 && source.StartsWith('0') && Uni.AllAsciiDigits(source)
            ? FallbackReason.LeadingZeroes
            : category switch
            {
                IssueCategory.Ambiguous => FallbackReason.MissingIntent,
                IssueCategory.InvalidExpression => FallbackReason.InvalidForm,
                IssueCategory.ProtectedIdentifier => FallbackReason.ProtectedIdentifier,
                IssueCategory.Unsupported => FallbackReason.UnsupportedForm,
                _ => FallbackReason.UnapprovedAbbreviation,
            };
        return new FallbackRequest(category, @class, reason, number, temporal, date, time, literal);
    }

    public static FallbackRequest Symbols() =>
        new(null, FallbackClass.Symbol, FallbackReason.UnhandledSymbol, literal: LetterReading.Spell);

    public static FallbackRequest Numeric(string source, NumericFailure failure) =>
        Primary(source, FallbackClass.Number, failure.Category, number: failure.Preference,
            literal: LetterReading.PreserveWords);

    public static FallbackRequest Temporal(string source, FallbackClass @class, TemporalFailure failure) =>
        failure.Preference is { } preference
            ? Primary(source, @class, failure.Category, temporal: preference)
            : Unresolved(source, @class, failure.Category);

    public (string Text, FallbackStrategy Strategy) Render(string source, int maximum, WorkControl control)
    {
        var output = new FallbackOutput(maximum, control);
        FallbackStrategy strategy;
        switch (number)
        {
            case NumberPreference preferred:
                output.Word(preferred.Number.Render().IntoText());
                strategy = FallbackStrategy.PreferredNumber;
                break;
            case SentenceNumberPreference sentence:
                output.Word(Numerals.NumberSpoken(sentence.Number).IntoText());
                output.Append(".");
                strategy = FallbackStrategy.PreferredNumber;
                break;
            default:
                if (date != null)
                {
                    output.Word(date.Render());
                    strategy = FallbackStrategy.SurfaceDate;
                }
                else if (temporal is DatePreference datePreference)
                {
                    output.Word(Verbalize.Render(new DateValue(datePreference.Date, datePreference.Locative)).Text);
                    strategy = FallbackStrategy.PreferredDate;
                }
                else if (temporal is TimePreference timePreference)
                {
                    output.Word(Verbalize.Render(new TimeValue(timePreference.Clock, timePreference.Locative)).Text);
                    strategy = FallbackStrategy.PreferredTime;
                }
                else if (time != null)
                {
                    output.Word(time.Render());
                    strategy = FallbackStrategy.SurfaceTime;
                }
                else
                {
                    strategy = Literal.Render(source, literal, output)
                        ? FallbackStrategy.UnicodeCodePoint
                        : FallbackStrategy.Literal;
                }
                break;
        }
        var text = output.Finish();
        if (text.Trim().Length == 0)
        {
            throw NormalizeException.Of(NormalizeErrorKind.Internal);
        }
        return (text, strategy);
    }

    public static bool NeedsReading(string grapheme, bool withinWord)
    {
        foreach (var scalar in grapheme.EnumerateRunes())
        {
            if (Spelling.SymbolName(scalar.Value) != null
                && !Spelling.ProsePunctuation(scalar.Value)
                && !(withinWord && Uni.IsAlphabetic(scalar)))
            {
                return true;
            }
        }
        var hasLetter = Uni.Any(grapheme, Uni.IsAlphabetic);
        return Uni.Any(grapheme, scalar =>
            !Uni.IsAlphabetic(scalar)
            && !(hasLetter && Uni.IsCombiningMark(scalar))
            && !Uni.IsWhitespace(scalar)
            && !Spelling.ProsePunctuation(scalar.Value));
    }

    static readonly string[] Emoticons = { ":D", ":)", ":(", ";)", ":P", "<3" };

    public static int? EmoticonLength(string source)
    {
        foreach (var emoticon in Emoticons)
        {
            if (source.StartsOrd(emoticon))
            {
                return emoticon.Length;
            }
        }
        return null;
    }
}

/// <summary>src/fallback/output.rs — üst sınır UTF-8 bayt olarak uygulanır.</summary>
internal sealed class FallbackOutput
{
    readonly StringBuilder text = new();
    readonly int maximum;
    readonly WorkControl control;
    int bytes;
    bool endsWithWhitespace; // metnin son rune'u boşluk mu (Rust: text.ends_with(char::is_whitespace))

    public FallbackOutput(int maximum, WorkControl control)
    {
        this.maximum = maximum;
        this.control = control;
    }

    public void Append(string part)
    {
        control.Check();
        var length = Uni.Len8(part);
        if ((long)bytes + length > maximum)
        {
            throw NormalizeException.Limited(LimitKind.Result);
        }
        text.Append(part);
        bytes += length;
        if (part.Length > 0 && Uni.RuneBefore(part, part.Length) is { } last)
        {
            endsWithWhitespace = Uni.IsWhitespace(last);
        }
    }

    public void Word(string word)
    {
        if (text.Length > 0 && !endsWithWhitespace)
        {
            Append(" ");
        }
        Append(word);
    }

    public string Finish() => text.ToString();
}

/// <summary>src/fallback/surface.rs — written components, deliberately not a validated Gregorian Date.</summary>
internal sealed class DateSurface
{
    readonly ulong day;
    readonly string month;
    readonly ulong year;

    DateSurface(ulong day, string month, ulong year)
    {
        this.day = day;
        this.month = month;
        this.year = year;
    }

    public static DateSurface? Parse(string text)
    {
        string day, month, year;
        if (text.Contains('.'))
        {
            var parts = text.Split('.');
            if (parts.Length != 3)
            {
                return null;
            }
            (day, month, year) = (parts[0], parts[1], parts[2]);
        }
        else
        {
            var parts = text.Split('-');
            if (parts.Length != 3)
            {
                return null;
            }
            (year, month, day) = (parts[0], parts[1], parts[2]);
        }
        if (Uni.Len8(day) is < 1 or > 2
            || Uni.Len8(month) is < 1 or > 2
            || Uni.Len8(year) != 4
            || !(Uni.AllAsciiDigits(day) && Uni.AllAsciiDigits(month) && Uni.AllAsciiDigits(year)))
        {
            return null;
        }
        var name = Lexicon.MonthName(int.Parse(month, System.Globalization.CultureInfo.InvariantCulture));
        if (name == null)
        {
            return null;
        }
        return new DateSurface(ulong.Parse(day, System.Globalization.CultureInfo.InvariantCulture), name,
            ulong.Parse(year, System.Globalization.CultureInfo.InvariantCulture));
    }

    public string Render() =>
        $"{Numerals.Cardinal(day).IntoText()} {month} {Numerals.Cardinal(year).IntoText()}";
}

/// <summary>Two source components, without asserting that they are a valid clock.</summary>
internal sealed class TimeSurface
{
    readonly ulong hour;
    readonly ulong minute;

    TimeSurface(ulong hour, ulong minute)
    {
        this.hour = hour;
        this.minute = minute;
    }

    public static TimeSurface? Parse(string text)
    {
        if (!Uni.SplitOnce(text, new[] { ':', '.' }, out var hour, out var minute))
        {
            return null;
        }
        if (Uni.Len8(hour) is < 1 or > 2 || Uni.Len8(minute) != 2
                                          || !(Uni.AllAsciiDigits(hour) && Uni.AllAsciiDigits(minute)))
        {
            return null;
        }
        return new TimeSurface(ulong.Parse(hour, System.Globalization.CultureInfo.InvariantCulture),
            ulong.Parse(minute, System.Globalization.CultureInfo.InvariantCulture));
    }

    public string Render() => $"{Numerals.Cardinal(hour).IntoText()} {Numerals.Cardinal(minute).IntoText()}";
}
