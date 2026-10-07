// src/classify/readers/lexical.rs ve electronic.rs portu
namespace NormalizerTr;

internal static class LexicalReader
{
    public static Attempt? Read(ReadContext ctx, int index)
    {
        var tokens = ctx.Tokens;
        var source = tokens[index].Text;
        if (source == "&")
        {
            return Attempt.Ok(new SymbolValue("ve"), index);
        }
        if (source.StartsWith('#'))
        {
            return Electronic.Hashtag(source) is { } hashtag
                ? Attempt.Ok(new SymbolValue(hashtag), index)
                : Attempt.Err(IssueCategory.Unsupported, index);
        }
        var labelBeforeNumber = index + 1 < tokens.Count
                                && NumericText.Label(source)
                                && Uni.AnyAsciiDigit(tokens[index + 1].Text);
        if (!labelBeforeNumber && NumericText.LexicalReading(source) is { } reading)
        {
            return reading.IsOk
                ? Attempt.Ok(new LexicalValue(reading.Value.Entry, reading.Value.Case), index)
                : Attempt.Err(reading.Error, index);
        }
        return null;
    }
}

internal static class ElectronicReader
{
    public static Attempt? Whole(ReadContext ctx, int index)
    {
        var source = ctx.Tokens[index].Text;
        if (Electronic.LooksLike(source))
        {
            return Electronic.Parse(source, false) is { } address
                ? Attempt.Ok(new ElectronicValue(address), index)
                : Attempt.Err(IssueCategory.Unsupported, index);
        }
        return null;
    }

    public static Attempt? Contextual(ReadContext ctx, int index)
    {
        var source = ctx.Tokens[index].Text;
        if (source.Contains('.') && ctx.Cue(index, new[] { "web", "site" }))
        {
            return Electronic.Parse(source, true) is { } address
                ? Attempt.Ok(new ElectronicValue(address), index)
                : Attempt.Err(IssueCategory.Unsupported, index);
        }
        if (source.Contains('.')
            && !Uni.Any(source, Uni.IsNumeric)
            && source.Split('.').All(part =>
                part.Length > 0 && Uni.All(part, c => Uni.IsAlphanumeric(c) || c.Value == '-')))
        {
            return Attempt.Err(
                Electronic.Parse(source, true) != null ? IssueCategory.Ambiguous : IssueCategory.Unsupported,
                index);
        }
        return null;
    }
}
