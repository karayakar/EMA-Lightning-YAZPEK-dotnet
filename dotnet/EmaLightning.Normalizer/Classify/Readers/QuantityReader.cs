// src/classify/readers/quantities.rs portu
namespace NormalizerTr;

internal static class QuantityReader
{
    const string CurrencySymbols = "₺$€£";

    public static Attempt? Read(ReadContext ctx, int index)
    {
        var text = ctx.Text;
        var tokens = ctx.Tokens;
        var token = tokens[index];
        var source = token.Text;
        bool LabelTail(int at, int after) =>
            at < tokens.Count
            && NumericText.Label(tokens[at].Text)
            && Scan.WhitespaceBetween(text, tokens[after].Range.End, tokens[at].Range.Start);

        // Prefix/suffix symbols are whole quantities; no fragment fallback for malformed values.
        if (source.Length > 0 && CurrencySymbols.Contains(source[0]))
        {
            var ch = source[0];
            if (QuantityMathEnd(text, tokens, index) is { } mathEnd)
            {
                return Attempt.Err(IssueCategory.Unsupported, mathEnd);
            }
            if (LabelTail(index + 1, index))
            {
                return Attempt.Err(IssueCategory.InvalidExpression, index + 1);
            }
            string number;
            string? @case;
            if (!NumericText.SplitSuffix(source[1..], out number, out @case))
            {
                number = "";
                @case = null;
            }
            var label = @case == null ? ch.ToString() : $"{ch}'{@case}";
            return Quantity.Parse(number, label) is { } quantity
                ? Attempt.Ok(new QuantityValue(quantity), index)
                : Attempt.Err(IssueCategory.InvalidExpression, index);
        }
        if (!NumericText.SplitSuffix(source, out var symbolBase, out var symbolCase))
        {
            symbolBase = source;
            symbolCase = null;
        }
        if (symbolBase.Length > 0 && CurrencySymbols.Contains(symbolBase[^1]))
        {
            var ch = symbolBase[^1];
            if (QuantityMathEnd(text, tokens, index) is { } mathEnd)
            {
                return Attempt.Err(IssueCategory.Unsupported, mathEnd);
            }
            if (LabelTail(index + 1, index))
            {
                return Attempt.Err(IssueCategory.InvalidExpression, index + 1);
            }
            var label = symbolCase == null ? ch.ToString() : $"{ch}'{symbolCase}";
            return Quantity.Parse(symbolBase[..^1], label) is { } quantity
                ? Attempt.Ok(new QuantityValue(quantity), index)
                : Attempt.Err(IssueCategory.InvalidExpression, index);
        }
        if (index + 1 >= tokens.Count)
        {
            return null;
        }
        var next = tokens[index + 1];
        if (!Scan.WhitespaceBetween(text, token.Range.End, next.Range.Start))
        {
            return null;
        }
        var hasDigit = Uni.AnyAsciiDigit(source);
        if (hasDigit
            && source.IndexOfAny(new[] { '-', '–' }) >= 0
            && (Lexicon.Unit(next.Text) != null || NumericText.IsCountNoun(next.Text)))
        {
            var end = index + 1;
            if (QuantityMathEnd(text, tokens, end) is { } mathEnd)
            {
                return Attempt.Err(IssueCategory.Unsupported, mathEnd);
            }
            return NumericRange.Parse(source, next.Text) is { } range
                ? Attempt.Ok(new RangeValue(range), end)
                : Attempt.Err(IssueCategory.InvalidExpression, end);
        }
        if (NumericText.Label(next.Text) && hasDigit)
        {
            if (QuantityMathEnd(text, tokens, index + 1) is { } mathEnd)
            {
                return Attempt.Err(IssueCategory.Unsupported, mathEnd);
            }
            if (LabelTail(index + 2, index + 1))
            {
                return Attempt.Err(IssueCategory.InvalidExpression, index + 2);
            }
            return Quantity.Parse(source, next.Text) is { } quantity
                ? Attempt.Ok(new QuantityValue(quantity), index + 1)
                : Attempt.Err(IssueCategory.InvalidExpression, index + 1);
        }
        if (hasDigit && Lexicon.Unit(Uni.AsciiLower(next.Text)) != null)
        {
            return Attempt.Err(IssueCategory.Unsupported, index + 1);
        }
        if (hasDigit
            && next.Text.Length > 0 && Uni.IsAlphabetic(Uni.FirstRune(next.Text))
            && (next.Text.IndexOfAny(new[] { '/', '^', '²', '³' }) >= 0
                || next.Text is "Μg" or "μG" or "µG" or "ug" or "oz" or "cl" or "dl" or "ms"))
        {
            return Attempt.Err(IssueCategory.Unsupported, index + 1);
        }
        if (NumericText.Label(source) && Uni.AnyAsciiDigit(next.Text))
        {
            return Quantity.Parse(next.Text, source) is { } quantity
                ? Attempt.Ok(new QuantityValue(quantity), index + 1)
                : Attempt.Err(IssueCategory.InvalidExpression, index + 1);
        }
        return null;
    }

    static int? QuantityMathEnd(string text, List<Token> tokens, int end)
    {
        var initial = end;
        while (end + 2 < tokens.Count)
        {
            var @operator = tokens[end + 1];
            var number = tokens[end + 2];
            if (!Scan.MathOperator(@operator.Text)
                || !Uni.AnyAsciiDigit(number.Text)
                || !Scan.WhitespaceBetween(text, tokens[end].Range.End, @operator.Range.Start)
                || !Scan.WhitespaceBetween(text, @operator.Range.End, number.Range.Start))
            {
                break;
            }
            end += 2;
            if (end + 1 < tokens.Count
                && NumericText.Label(tokens[end + 1].Text)
                && Scan.WhitespaceBetween(text, tokens[end].Range.End, tokens[end + 1].Range.Start))
            {
                end += 1;
            }
        }
        return end > initial ? end : null;
    }
}
