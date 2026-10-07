// src/classify/readers/numeric.rs portu
namespace NormalizerTr;

internal static class NumericReader
{
    public static (Attempt Attempt, FallbackClass Class)? Read(ReadContext ctx, int index)
    {
        var text = ctx.Text;
        var tokens = ctx.Tokens;
        var token = tokens[index];
        var source = token.Text;
        if (NumericText.Percent(source) is { } percent)
        {
            var attempt = percent.IsOk
                ? Attempt.Ok(new PercentValue(percent.Value.Number, percent.Value.Case), index)
                : Attempt.Err(percent.Error, index);
            return (attempt, FallbackClass.Percent);
        }
        var @base = Uni.Head(source);
        if ((@base.EndsWith('.') || source.IndexOfAny(Uni.Apostrophes) >= 0)
            && Numeric.Parse(source, false) is { } number)
        {
            return (Attempt.Ok(new NumericValue(number), index), FallbackClass.Number);
        }
        var romanBase = Uni.Head(source).TrimEnd('.');
        if (romanBase.Length > 0 && Uni.AllChars(romanBase, c => c is 'I' or 'V' or 'X' or 'L' or 'C' or 'D' or 'M'))
        {
            var contextual = false;
            if (source.EndsWith('.') && index + 1 < tokens.Count)
            {
                var next = tokens[index + 1];
                if (Scan.WhitespaceBetween(text, token.Range.End, next.Range.Start))
                {
                    var key = Lexicon.LookupKey(next.Text);
                    contextual = key == "yüzyıl"
                                 || (key == "dünya"
                                     && index + 2 < tokens.Count
                                     && Lexicon.LookupKey(tokens[index + 2].Text) == "savaşı"
                                     && Scan.WhitespaceBetween(text, next.Range.End, tokens[index + 2].Range.Start));
                }
            }
            Attempt attempt;
            if (contextual)
            {
                attempt = Numeric.Roman(source) is { } roman
                    ? Attempt.Ok(new RomanValue(roman), index)
                    : Attempt.Err(IssueCategory.InvalidExpression, index);
            }
            else
            {
                attempt = Attempt.Err(IssueCategory.Ambiguous, index);
            }
            return (attempt, FallbackClass.Roman);
        }
        return null;
    }
}
