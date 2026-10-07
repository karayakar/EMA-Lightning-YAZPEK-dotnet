// src/verbalize.rs portu
namespace NormalizerTr;

internal static class Verbalize
{
    public static Spoken DateSpoken(Date date)
    {
        var year = Numerals.Cardinal((ulong)date.Year);
        var prefix = $"{Numerals.Cardinal((ulong)date.Day).IntoText()} {Lexicon.Months[date.Month - 1]} ";
        year.Prefix(prefix);
        return year;
    }

    public static Spoken TimeSpoken(Clock time)
    {
        var hour = Numerals.Cardinal((ulong)time.Hour);
        if (time.Minute == 0)
        {
            return hour;
        }
        var minute = Numerals.Cardinal((ulong)time.Minute);
        minute.Prefix($"{hour.IntoText()} ");
        return minute;
    }

    public static (SegmentKind Kind, string RuleId, string Text) Render(Value value)
    {
        switch (value)
        {
            case NumericValue n:
                return (n.Number.Kind, "number", n.Number.Render().IntoText());
            case DigitsValue d:
                return (SegmentKind.Digits, "digits.hint", Numerals.DigitsText(d.Text));
            case DateValue d:
            {
                var spoken = DateSpoken(d.Date);
                if (d.Locative)
                {
                    spoken.Inflect(Inflection.Locative);
                }
                return (SegmentKind.Date, "date.gregorian", spoken.IntoText());
            }
            case TimeValue t:
            {
                var spoken = TimeSpoken(t.Clock);
                if (t.Locative)
                {
                    spoken.Inflect(Inflection.Locative);
                }
                return (SegmentKind.Time, "time.digital", spoken.IntoText());
            }
            case PercentValue p:
            {
                var spoken = Numerals.NumberSpoken(p.Number);
                if (p.Case is { } c)
                {
                    spoken.Inflect(c);
                }
                spoken.Prefix("yüzde ");
                return (SegmentKind.Percent, "percent", spoken.IntoText());
            }
            case QuantityValue q:
                return (q.Quantity.IsMoney ? SegmentKind.Money : SegmentKind.Unit, "quantity",
                    q.Quantity.Render().IntoText());
            case LexicalValue l:
            {
                var spoken = Spoken.Lexical(l.Entry.Output, l.Entry.Target);
                if (l.Case is { } c)
                {
                    spoken.Inflect(c);
                }
                return (SegmentKind.Abbreviation, "abbreviation", spoken.IntoText());
            }
            case RangeValue r:
                return (SegmentKind.Range, "range.context", r.Range.Render());
            case TelephoneValue t:
                return (SegmentKind.Telephone, "telephone.tr", t.Phone.Render());
            case IbanValue i:
                return (SegmentKind.Iban, "iban.tr.mod97", i.Iban.Render());
            case RomanValue r:
                return (SegmentKind.Roman, "roman.canonical", r.Number.Render().IntoText());
            case ElectronicValue e:
                return (SegmentKind.Electronic, "electronic.ascii", e.Address.Render());
            case SymbolValue s:
                return (SegmentKind.Symbol, "symbol.prose", s.Text);
            default:
                throw NormalizeException.Of(NormalizeErrorKind.Internal);
        }
    }
}
