// src/classify/scan.rs portu
namespace NormalizerTr;

internal readonly record struct Token(Rng Range, string Text);

internal static class Scan
{
    public static bool Overlaps(Rng a, Rng b) => a.Start < b.End && b.Start < a.End;

    public static bool WhitespaceBetween(string text, int start, int end) =>
        start < end && Uni.All(text[start..end], Uni.IsWhitespace);

    public static bool CueWhitespace(string text, int start, int end) =>
        start == end || WhitespaceBetween(text, start, end);

    static bool Delimiter(char ch) =>
        ch is ';' or '!' or '?' or '(' or ')' or '{' or '}' or '[' or ']' or '«' or '»' or '"';

    static Rng? TrimmedRange(string text, int offset)
    {
        var leading = Uni.TrimStartBy(text, ch => Delimiter(ch) || ch is ',' or '…');
        var body = Uni.TrimEndBy(leading, ch => Delimiter(ch) || ch == '…');
        if (body.EndsWith(',') && !body.EndsOrd(",,"))
        {
            body = body[..^1];
        }
        if (body.EndsWith('.')
            && !body.EndsOrd("..")
            && body != "Dr."
            && !Uni.AllAsciiDigits(body[..^1]))
        {
            body = body[..^1];
        }
        var start = offset + text.Length - leading.Length;
        return body.Length > 0 ? new Rng(start, start + body.Length) : null;
    }

    static Rng? ExpressionRange(string text, int offset)
    {
        if (text.StartsWith('"') && text.Contains('@'))
        {
            var quoted = text.TrimEnd(',', ';', '!');
            return new Rng(offset, offset + quoted.Length);
        }
        var leading = text.TrimStart('(', '[', '{', '«', '"');
        if (Electronic.LooksLike(leading))
        {
            var body = leading.TrimEnd('.', ',', ';', '!', '»', '"');
            while (body.EndsWith(')') && Uni.Count(body, ')') > Uni.Count(body, '('))
            {
                body = body[..^1];
            }
            body = body.TrimEnd(']', '}');
            var start = offset + text.Length - leading.Length;
            return body.Length > 0 ? new Rng(start, start + body.Length) : null;
        }
        if (TrimmedRange(text, offset) is not { } range)
        {
            return null;
        }
        var raw = text[(range.Start - offset)..];
        var without = Uni.TrimEndBy(raw, c => Delimiter(c) || c is ',' or '…');
        var @base = Uni.Head(without);
        var roman = @base.TrimEnd('.');
        var retain = LexicalPeriod(without) || (roman.Length > 0 && Uni.AllChars(roman, IsRomanLetter));
        return retain ? new Rng(range.Start, range.Start + without.Length) : range;
    }

    static bool IsRomanLetter(char c) => c is 'I' or 'V' or 'X' or 'L' or 'C' or 'D' or 'M';

    static bool LexicalPeriod(string text) => Lexicon.Abbreviation(Uni.Head(text)) != null;

    static bool NumericParenthesisCompound(string raw)
    {
        var body = raw.StartsWith('(') && raw.EndsWith(')') && raw.Length >= 2 ? raw[1..^1] : raw;
        if (body.IndexOfAny(new[] { '(', ')' }) < 0 || Uni.Any(body, Uni.IsAlphabetic))
        {
            return false;
        }
        var runs = 0;
        var inRun = false;
        foreach (var r in body.EnumerateRunes())
        {
            if (Uni.IsAsciiDigit(r))
            {
                if (!inRun)
                {
                    runs++;
                    inRun = true;
                }
            }
            else
            {
                inRun = false;
            }
        }
        return runs > 1;
    }

    public static List<Token> Tokens(SourceMap source, WorkControl control)
    {
        var text = source.Text;
        var tokens = new List<Token>();
        var skipUntil = 0;
        foreach (System.Text.RegularExpressions.Match matched in Resources.Tokens.Matches(text))
        {
            control.Check();
            var matchedStart = matched.Index;
            var matchedEnd = matched.Index + matched.Length;
            if (matchedStart < skipUntil)
            {
                continue;
            }
            var raw = matched.Value;
            if (raw.StartsWith('"'))
            {
                var address = Resources.QuotedEmail.Match(text, matchedStart);
                if (address.Success && address.Index == matchedStart)
                {
                    var body = address.Value.TrimEnd(',', ';', '!');
                    AppendToken(tokens, source, new Rng(address.Index, address.Index + body.Length));
                    skipUntil = address.Index + address.Length;
                    continue;
                }
            }
            if (NumericParenthesisCompound(raw))
            {
                AppendToken(tokens, source, new Rng(matchedStart, matchedEnd));
                continue;
            }
            if (ExpressionRange(raw, matchedStart) is not { } range)
            {
                continue;
            }
            var expression = text[range.Start..range.End];
            if (Electronic.LooksLike(expression)
                || LexicalPeriod(expression)
                || Uni.AllChars(expression.TrimEnd('.'), IsRomanLetter))
            {
                AppendToken(tokens, source, range);
                continue;
            }
            int? prefix = CueContext.InlinePrefix(expression);
            if (prefix == null
                && expression.StartsWith(':')
                && expression.Length > 1 && Uni.IsAsciiDigit(expression[1])
                && tokens.Count > 0
                && !tokens[^1].Text.Contains(':')
                && CueContext.IsCueWord(tokens[^1].Text)
                && WhitespaceBetween(text, tokens[^1].Range.End, range.Start))
            {
                prefix = 1;
            }
            if (prefix is { } p)
            {
                var bodyStart = range.Start + p;
                var bodyRange = TrimmedRange(text[bodyStart..matchedEnd], bodyStart)
                                ?? throw NormalizeException.Of(NormalizeErrorKind.Internal);
                AppendToken(tokens, source, new Rng(range.Start, range.Start + p));
                AppendToken(tokens, source, bodyRange);
                continue;
            }
            // Identifier punctuation (including URL queries) must not split the token.
            if (Identifier(expression))
            {
                AppendToken(tokens, source, range);
                continue;
            }
            var start = 0;
            for (var offset = 0; offset < raw.Length; offset++)
            {
                if (!Delimiter(raw[offset]))
                {
                    continue;
                }
                if (TrimmedRange(raw[start..offset], matchedStart + start) is { } piece)
                {
                    AppendToken(tokens, source, piece);
                }
                start = offset + 1;
            }
            if (TrimmedRange(raw[start..], matchedStart + start) is { } last)
            {
                AppendToken(tokens, source, last);
            }
        }
        return tokens;
    }

    static void AppendToken(List<Token> tokens, SourceMap source, Rng range)
    {
        var covered = source.Cover(range);
        tokens.Add(new Token(covered, source.Text[covered.Start..covered.End]));
    }

    public static List<Rng> Phones(string text, List<Token> tokens, WorkControl control)
    {
        var phones = new List<Rng>();
        foreach (System.Text.RegularExpressions.Match matched in Resources.PhoneLike.Matches(text))
        {
            control.Check();
            var range = new Rng(matched.Index, matched.Index + matched.Length);
            if (Seq.BinarySearch(tokens, range.Start, t => t.Range.Start) >= 0
                && Seq.BinarySearch(tokens, range.End, t => t.Range.End) >= 0)
            {
                phones.Add(range);
            }
        }
        return phones;
    }

    public static int? SpacedCompound(string text, List<Token> tokens, int index)
    {
        if (!Uni.AnyAsciiDigit(tokens[index].Text))
        {
            return null;
        }
        var end = index;
        while (end + 2 < tokens.Count)
        {
            var @operator = tokens[end + 1];
            var number = tokens[end + 2];
            if (!MathOperator(@operator.Text)
                || !Uni.AnyAsciiDigit(number.Text)
                || !WhitespaceBetween(text, tokens[end].Range.End, @operator.Range.Start)
                || !WhitespaceBetween(text, @operator.Range.End, number.Range.Start))
            {
                break;
            }
            end += 2;
        }
        return end > index ? end : null;
    }

    public static bool MathOperator(string text) => text is "/" or "-" or "–" or "—" or ":" or "x" or "×" or "^"
        or "+" or "=" or "*" or "÷" or "<" or ">" or "≤" or "≥" or "≈" or "±";

    public static bool Identifier(string text)
    {
        if (text.Contains('@') || text.Contains("://", StringComparison.Ordinal) || text.StartsOrd("www."))
        {
            return true;
        }
        var @base = Uni.Head(text);
        var letters = Uni.Any(@base, c => Uni.IsAlphabetic(c) && !Uni.IsNumeric(c));
        var digits = Uni.Any(@base, Uni.IsNumeric);
        return digits && (letters || @base.Contains('_'));
    }

    public static bool UnknownAbbreviation(string text)
    {
        var @base = Uni.Head(text);
        var letters = @base.EnumerateRunes().Where(Uni.IsAlphabetic).ToList();
        if (letters.Count < 2)
        {
            return false;
        }
        return letters.All(Uni.IsUppercase);
    }
}
