// src/classify/readers/identifiers.rs portu
namespace NormalizerTr;

internal static class IdentifierReader
{
    static bool FourDigits(string text) => Uni.Len8(text) == 4 && Uni.AllAsciiDigits(text);

    public static Attempt? Read(ReadContext ctx, int index)
    {
        var text = ctx.Text;
        var tokens = ctx.Tokens;
        var token = tokens[index];
        var source = token.Text;
        var length = Uni.Len8(source);
        string? At(int i) => i < tokens.Count ? tokens[i].Text : null;

        var numericGroupContext = At(index + 1) is { } g1 && FourDigits(g1) && At(index + 2) is { } g2 && FourDigits(g2);
        var ascii = Uni.IsAscii(source);
        var ibanLike =
            (length >= 4
             && ascii
             && Uni.IsAsciiAlpha(source[0]) && Uni.IsAsciiAlpha(source[1])
             && Uni.AllChars(source[2..], Uni.IsAsciiAlnum)
             && (((source.StartsOrd("TR") || source.StartsOrd("tr"))
                  && (Uni.AllAsciiDigits(source[2..])
                      || length == 26
                      || (length == 4 && (Uni.AnyAsciiDigit(source[2..]) || numericGroupContext))))
                 || (length == 4 && Uni.AllAsciiDigits(source[2..]) && numericGroupContext)))
            || (source == "TR" && At(index + 1) is { } pair && Uni.Len8(pair) == 2 && Uni.AllAsciiDigits(pair));
        if (ibanLike)
        {
            var end = IbanEnd(text, tokens, index);
            var whole = text[token.Range.Start..tokens[end].Range.End];
            return Iban.Parse(whole) is { } iban
                ? Attempt.Ok(new IbanValue(iban), end)
                : Attempt.Err(IssueCategory.ProtectedIdentifier, end);
        }
        var zeroGroupSignal = source == "0"
                              && At(index + 1) is { } z1 && Uni.Len8(z1) == 3 && Uni.AllAsciiDigits(z1)
                              && At(index + 2) is { } z2 && Uni.Len8(z2) >= 2 && Uni.AllAsciiDigits(z2);
        var phoneCue = ctx.Cue(index, new[] { "telefon", "tel" });
        var telephoneCue = Uni.AnyAsciiDigit(source) && phoneCue;
        var quantityContext = telephoneCue
                              && At(index + 1) is { } q
                              && (NumericText.Label(q) || NumericText.IsCountNoun(q));
        if (source.StartsOrd("+90")
            || (source.StartsWith('0') && source != "0" && Uni.AnyAsciiDigit(source))
            || zeroGroupSignal
            || (source == "0" && phoneCue)
            || (telephoneCue && !quantityContext && Uni.AllAsciiDigits(source))
            || (length == 10 && Uni.AllAsciiDigits(source) && phoneCue))
        {
            var end = GroupEnd(text, tokens, index);
            var whole = text[token.Range.Start..tokens[end].Range.End];
            var digits = whole.Count(Uni.IsAsciiDigit);
            var dash = source.IndexOfAny(new[] { '-', '(' });
            var writtenPrefix = dash < 0 ? source : source[..dash];
            var nationalShape = source.StartsWith('0')
                                && ((length == 4 && Uni.AllAsciiDigits(source))
                                    || (Uni.Len8(writtenPrefix) == 4 && Uni.AllAsciiDigits(writtenPrefix))
                                    || (length >= 10 && Uni.AllAsciiDigits(source))
                                    || zeroGroupSignal
                                    || (source.StartsOrd("0-") && digits == 11)
                                    || telephoneCue);
            var contextualRange = At(index + 1) is { } r && NumericRange.Parse(source, r) != null;
            if (!contextualRange
                && !quantityContext
                && ((source.StartsOrd("+90") && (end > index || digits == 12)) || nationalShape || telephoneCue))
            {
                return Telephone.Parse(whole, telephoneCue) is { } phone
                    ? Attempt.Ok(new TelephoneValue(phone), end)
                    : Attempt.Err(IssueCategory.InvalidExpression, end);
            }
        }
        return null;
    }

    static bool NumberGroup(string token) =>
        token.Length > 0 && Uni.AllChars(token, ch => Uni.IsAsciiDigit(ch) || ch is '+' or '-' or '(' or ')');

    static int GroupEnd(string text, List<Token> tokens, int index)
    {
        var end = index;
        while (end + 1 < tokens.Count)
        {
            var next = tokens[end + 1];
            var groupLike = NumberGroup(next.Text)
                            || (Uni.Len8(next.Text) <= 4
                                && next.Text.Length > 0 && Uni.IsAsciiDigit(next.Text[0])
                                && Uni.AllChars(next.Text, Uni.IsAsciiAlnum));
            var gap = text[tokens[end].Range.End..next.Range.Start];
            if (!groupLike || !Uni.All(gap, c => Uni.IsWhitespace(c) || c.Value is '(' or ')' or '-'))
            {
                break;
            }
            end++;
        }
        return end;
    }

    static int IbanEnd(string text, List<Token> tokens, int index)
    {
        var end = index;
        var characters = Uni.Len8(tokens[index].Text);
        while (end + 1 < tokens.Count)
        {
            var next = tokens[end + 1];
            if (Lexicon.Abbreviation(next.Text) != null
                || NumericText.Label(next.Text)
                || (Uni.AllAsciiDigits(next.Text)
                    && end + 2 < tokens.Count
                    && NumericText.Label(tokens[end + 2].Text)
                    && Scan.WhitespaceBetween(text, next.Range.End, tokens[end + 2].Range.Start)))
            {
                break;
            }
            if (!Scan.WhitespaceBetween(text, tokens[end].Range.End, next.Range.Start)
                || next.Text.Length == 0
                || !Uni.AllChars(next.Text, b => Uni.IsAsciiDigit(b) || Uni.IsAsciiUpper(b))
                || (!Uni.AnyAsciiDigit(next.Text) && Uni.Len8(next.Text) > 4)
                || (characters >= 26 && !Uni.AllAsciiDigits(next.Text)))
            {
                break;
            }
            characters += Uni.Len8(next.Text);
            end++;
        }
        return end;
    }
}
