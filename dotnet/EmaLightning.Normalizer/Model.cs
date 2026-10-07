// src/model.rs portu
using System.Globalization;

namespace NormalizerTr;

internal readonly record struct Date(int Day, int Month, int Year, bool Dotted)
{
    public static Date? Parse(string text)
    {
        var dotted = text.Contains('.');
        var parts = text.Split(dotted ? '.' : '-');
        if (parts.Length != 3)
        {
            return null;
        }
        var (day, month, year) = dotted ? (parts[0], parts[1], parts[2]) : (parts[2], parts[1], parts[0]);
        if (Uni.Len8(year) != 4
            || Uni.Len8(day) is < 1 or > 2
            || Uni.Len8(month) is < 1 or > 2
            || (!dotted && (Uni.Len8(day) != 2 || Uni.Len8(month) != 2))
            || !(Uni.AllAsciiDigits(day) && Uni.AllAsciiDigits(month) && Uni.AllAsciiDigits(year)))
        {
            return null;
        }
        var d = int.Parse(day, NumberStyles.None, CultureInfo.InvariantCulture);
        var m = int.Parse(month, NumberStyles.None, CultureInfo.InvariantCulture);
        var y = int.Parse(year, NumberStyles.None, CultureInfo.InvariantCulture);
        var leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
        int days;
        switch (m)
        {
            case 2:
                days = leap ? 29 : 28;
                break;
            case 4 or 6 or 9 or 11:
                days = 30;
                break;
            case 1 or 3 or 5 or 7 or 8 or 10 or 12:
                days = 31;
                break;
            default:
                return null;
        }
        if (y == 0 || d == 0 || d > days)
        {
            return null;
        }
        return new Date(d, m, y, dotted);
    }
}

internal readonly record struct Clock(int Hour, int Minute)
{
    public static Clock? Parse(string text)
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
        var h = int.Parse(hour, NumberStyles.None, CultureInfo.InvariantCulture);
        var m = int.Parse(minute, NumberStyles.None, CultureInfo.InvariantCulture);
        if (h > 23 || m > 59)
        {
            return null;
        }
        return new Clock(h, m);
    }
}

internal abstract record Value;

internal sealed record NumericValue(Numeric Number) : Value;

internal sealed record DigitsValue(string Text) : Value;

internal sealed record PercentValue(Number Number, Inflection? Case) : Value;

internal sealed record DateValue(Date Date, bool Locative) : Value;

internal sealed record TimeValue(Clock Clock, bool Locative) : Value;

internal sealed record QuantityValue(Quantity Quantity) : Value;

internal sealed record LexicalValue(Lexeme Entry, Inflection? Case) : Value;

internal sealed record RangeValue(NumericRange Range) : Value;

internal sealed record TelephoneValue(Telephone Phone) : Value;

internal sealed record IbanValue(Iban Iban) : Value;

internal sealed record RomanValue(Numeric Number) : Value;

internal sealed record ElectronicValue(Electronic Address) : Value;

internal sealed record SymbolValue(string Text) : Value;

internal abstract record TemporalPreference;

internal sealed record DatePreference(Date Date, bool Locative) : TemporalPreference;

internal sealed record TimePreference(Clock Clock, bool Locative) : TemporalPreference;

internal sealed record TemporalFailure(IssueCategory Category, TemporalPreference? Preference = null);
