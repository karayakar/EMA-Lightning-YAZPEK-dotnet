// src/classify/readers/mod.rs portu
namespace NormalizerTr;

internal readonly record struct Attempt(Res<Value, IssueCategory> Outcome, int End)
{
    public static Attempt Ok(Value value, int end) => new(Res<Value, IssueCategory>.Ok(value), end);
    public static Attempt Err(IssueCategory category, int end) => new(Res<Value, IssueCategory>.Err(category), end);
}

internal sealed class ReadContext
{
    public ReadContext(string text, List<Token> tokens)
    {
        Text = text;
        Tokens = tokens;
    }

    public string Text { get; }
    public List<Token> Tokens { get; }

    public bool Cue(int index, string[] allowed)
    {
        var i = index - 1;
        return i >= 0
               && Scan.CueWhitespace(Text, Tokens[i].Range.End, Tokens[index].Range.Start)
               && allowed.Contains(CueContext.CueKey(Tokens[i].Text));
    }

    public bool ContextualCueWord(int index)
    {
        var key = CueContext.CueKey(Tokens[index].Text);
        return CueContext.IsCueWord(Tokens[index].Text)
               || (key is "telefon" or "tel" or "web" or "site"
                   && index + 1 < Tokens.Count
                   && Scan.WhitespaceBetween(Text, Tokens[index].Range.End, Tokens[index + 1].Range.Start)
                   && (Tokens[index + 1].Text.Contains('.') || Uni.AnyAsciiDigit(Tokens[index + 1].Text)));
    }

    /// <summary>Anchor roles are recorded once after a successfully read contextual Roman.</summary>
    public int? RomanAnchorEnd(int index)
    {
        var token = Tokens[index];
        if (!token.Text.EndsWith('.') || index + 1 >= Tokens.Count)
        {
            return null;
        }
        var next = Tokens[index + 1];
        if (!Scan.WhitespaceBetween(Text, token.Range.End, next.Range.Start))
        {
            return null;
        }
        switch (Lexicon.LookupKey(next.Text))
        {
            case "yüzyıl":
                return index + 2;
            case "dünya":
                if (index + 2 < Tokens.Count)
                {
                    var last = Tokens[index + 2];
                    if (Lexicon.LookupKey(last.Text) == "savaşı"
                        && Scan.WhitespaceBetween(Text, next.Range.End, last.Range.Start))
                    {
                        return index + 3;
                    }
                }
                return null;
            default:
                return null;
        }
    }
}

internal static class Readers
{
    public static Value? Hint(string text, HintKind kind)
    {
        switch (kind)
        {
            case HintKind.Cardinal:
                return Numeric.CardinalHint(text) is { } cardinal ? new NumericValue(cardinal) : null;
            case HintKind.Digits:
                return NumericText.DigitsHint(text) is { } digits ? new DigitsValue(digits) : null;
            case HintKind.Date:
            {
                var date = Temporal.DateReading(text, true);
                return date.IsOk ? date.Value : null;
            }
            case HintKind.Time:
            {
                var time = Temporal.TimeReading(text, true);
                return time.IsOk ? time.Value : null;
            }
            case HintKind.Ordinal:
                return Numeric.Parse(text, true) is { } ordinal ? new NumericValue(ordinal) : null;
            case HintKind.Roman:
                return Numeric.Roman(text) is { } roman ? new RomanValue(roman) : null;
            case HintKind.Telephone:
                return Telephone.Parse(text, true) is { } phone ? new TelephoneValue(phone) : null;
            case HintKind.Electronic:
                return Electronic.Parse(text, true) is { } address ? new ElectronicValue(address) : null;
            default:
            {
                var space = text.LastIndexOf(' ');
                var body = space < 0 ? text : text[..space];
                var noun = space < 0 ? null : text[(space + 1)..];
                return NumericRange.Parse(body, noun) is { } range ? new RangeValue(range) : null;
            }
        }
    }

    /// <summary>Fixed precedence. Some(Err) seals a matched invalid span; only None continues.</summary>
    public static (Reading Reading, int End)? Read(ReadContext ctx, int index, Boundaries bounds, bool contextualRole)
    {
        (Reading, int) Annotate(Attempt attempt, FallbackClass @class)
        {
            var span = ctx.Text[ctx.Tokens[index].Range.Start..ctx.Tokens[attempt.End].Range.End];
            var annotated = attempt.Outcome.IsOk
                ? Reading.Of(attempt.Outcome.Value)
                : Reading.Of(FallbackRequest.Unresolved(span, @class, attempt.Outcome.Error));
            return (annotated, attempt.End);
        }

        if (ElectronicReader.Whole(ctx, index) is { } whole)
        {
            return Annotate(whole, FallbackClass.Electronic);
        }
        if (LexicalReader.Read(ctx, index) is { } lexical)
        {
            return Annotate(lexical, FallbackClass.Abbreviation);
        }
        if (ElectronicReader.Contextual(ctx, index) is { } contextual)
        {
            return Annotate(contextual, FallbackClass.Electronic);
        }
        if (IdentifierReader.Read(ctx, index) is { } identifier)
        {
            return Annotate(identifier, FallbackClass.Identifier);
        }
        if (NumericReader.Read(ctx, index) is { } numeric)
        {
            return Annotate(numeric.Attempt, numeric.Class);
        }
        if (QuantityReader.Read(ctx, index) is { } quantity)
        {
            return Annotate(quantity, FallbackClass.Quantity);
        }
        if (bounds.PhoneAt(ctx.Tokens[index].Range.Start) is { } phone)
        {
            return Annotate(Attempt.Err(IssueCategory.Unsupported, bounds.NextAt(phone.End) - 1),
                FallbackClass.Identifier);
        }
        if (Scan.SpacedCompound(ctx.Text, ctx.Tokens, index) is { } end)
        {
            return Annotate(Attempt.Err(IssueCategory.Unsupported, end), FallbackClass.Expression);
        }
        if (UnsupportedQuantity(ctx, index) is { } unsupported)
        {
            return Annotate(unsupported, FallbackClass.Quantity);
        }
        if (TokenReading(ctx, index, contextualRole) is { } reading)
        {
            return (reading, index);
        }
        return null;
    }

    static Attempt? UnsupportedQuantity(ReadContext ctx, int index)
    {
        var token = ctx.Tokens[index];
        var text = ctx.Text;
        Token? next = index + 1 < ctx.Tokens.Count ? ctx.Tokens[index + 1] : null;
        if (token.Text is "%" or "+" or "-" or "√" or "∛"
            && next is { } n1
            && Uni.Any(n1.Text, Uni.IsNumeric)
            && Scan.WhitespaceBetween(text, token.Range.End, n1.Range.Start))
        {
            return Attempt.Err(IssueCategory.Unsupported, index + 1);
        }
        if (NumericText.UnsupportedLabel(token.Text)
            && next is { } n2
            && Uni.AnyAsciiDigit(n2.Text)
            && Scan.WhitespaceBetween(text, token.Range.End, n2.Range.Start))
        {
            return Attempt.Err(IssueCategory.Unsupported, index + 1);
        }
        if (token.Text.StartsWith('¥') || token.Text.EndsWith('¥'))
        {
            return Attempt.Err(IssueCategory.Unsupported, index);
        }
        if (Uni.AnyAsciiDigit(token.Text)
            && next is { } n3
            && Boundaries.QuantityTail(n3.Text)
            && Scan.WhitespaceBetween(text, token.Range.End, n3.Range.Start))
        {
            return Attempt.Err(IssueCategory.Unsupported, index + 1);
        }
        return null;
    }

    static Reading? TokenReading(ReadContext ctx, int index, bool contextualRole)
    {
        var token = ctx.Tokens[index];
        if (Scan.Identifier(token.Text))
        {
            return Reading.Of(FallbackRequest.Unresolved(token.Text, FallbackClass.Identifier,
                IssueCategory.ProtectedIdentifier));
        }
        if (token.Text.IndexOfAny(new[] { ':', '.', '-' }) >= 0 && Uni.Any(token.Text, Uni.IsNumeric))
        {
            if (Temporal.Recognize(ctx.Text, ctx.Tokens, index) is { } recognized)
            {
                return recognized.Reading.IsOk
                    ? Reading.Of(recognized.Reading.Value)
                    : Reading.Of(FallbackRequest.Temporal(token.Text, recognized.Class, recognized.Reading.Error));
            }
            return AutomaticNumber(token.Text);
        }
        if (token.Text.StartsWith('%') || Uni.Any(token.Text, Uni.IsNumeric))
        {
            return AutomaticNumber(token.Text);
        }
        return Scan.UnknownAbbreviation(token.Text) && !contextualRole && !ctx.ContextualCueWord(index)
            ? Reading.Of(FallbackRequest.Unresolved(token.Text, FallbackClass.Abbreviation,
                IssueCategory.UnknownAbbreviation))
            : null;
    }

    static Reading AutomaticNumber(string source)
    {
        var value = Numeric.Automatic(source);
        return value.IsOk
            ? Reading.Of(new NumericValue(value.Value))
            : Reading.Of(FallbackRequest.Numeric(source, value.Error));
    }
}
