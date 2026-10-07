// src/fallback/literal.rs portu
using System.Globalization;
using System.Text;

namespace NormalizerTr;

internal static class Literal
{
    /// <summary>Ordered borrowed runs. Numeric interpretation never crosses a delimiter.</summary>
    public static bool Render(string source, LetterReading letters, FallbackOutput output)
    {
        var cursor = 0;
        var usedCodePoint = false;
        while (cursor < source.Length)
        {
            var remainder = source[cursor..];
            var first = Uni.FirstRune(remainder);
            if (Uni.IsWhitespace(first))
            {
                var end = Uni.Find(remainder, scalar => !Uni.IsWhitespace(scalar));
                output.Append(remainder[..end]);
                cursor += end;
                continue;
            }
            int length;
            if (Uni.IsAsciiDigit(first))
            {
                var notationLength = Uni.Find(remainder,
                    scalar => !Uni.IsAsciiDigit(scalar) && scalar.Value is not ('.' or ','));
                if (letters == LetterReading.PreserveWords
                    && Number.Parse(remainder[..notationLength]) is { } notation)
                {
                    output.Word(Numerals.NumberSpoken(notation).IntoText());
                    cursor += notationLength;
                    continue;
                }
                length = Uni.Find(remainder, scalar => !Uni.IsAsciiDigit(scalar));
                var digits = remainder[..length];
                if (letters == LetterReading.PreserveWords && Number.Parse(digits) is { } number)
                {
                    output.Word(Numerals.NumberSpoken(number).IntoText());
                }
                else
                {
                    EmitDigits(digits, output);
                }
            }
            else if (Uni.IsAlphabetic(first) && Spelling.SymbolName(first.Value) == null)
            {
                length = Uni.Find(remainder, scalar => !Uni.IsAlphabetic(scalar));
                usedCodePoint |= EmitWord(remainder[..length], letters, output);
            }
            else
            {
                length = Uni.GraphemeLength(remainder);
                usedCodePoint |= EmitSymbol(remainder[..length], output);
            }
            cursor += length;
        }
        return usedCodePoint;
    }

    static void EmitDigits(string digits, FallbackOutput output)
    {
        foreach (var digit in digits)
        {
            output.Word(Numerals.DigitName(digit) ?? throw NormalizeException.Of(NormalizeErrorKind.Internal));
        }
    }

    static bool EmitWord(string source, LetterReading reading, FallbackOutput output)
    {
        if (reading == LetterReading.Spell)
        {
            return EmitLetters(source, output);
        }
        Lexeme? label = Lexicon.Unit(source) ?? Lexicon.Abbreviation(source);
        if (label == null && Lexicon.ParseCurrency(source) is { } currency)
        {
            label = Lexicon.CurrencyLexeme(currency, source);
        }
        if (label is { } l)
        {
            output.Word(l.Output);
            return false;
        }
        if (Uni.All(source, Uni.IsUppercase))
        {
            return EmitLetters(source, output);
        }
        output.Word(source);
        return false;
    }

    static bool EmitLetters(string letters, FallbackOutput output)
    {
        var usedCodePoint = false;
        foreach (var letter in letters.EnumerateRunes())
        {
            if ((Spelling.LetterName(letter.Value) ?? Spelling.SymbolName(letter.Value)) is { } name)
            {
                output.Word(name);
            }
            else
            {
                CodePoint(letter, output);
                usedCodePoint = true;
            }
        }
        return usedCodePoint;
    }

    static bool EmitSymbol(string grapheme, FallbackOutput output)
    {
        var scalars = grapheme.EnumerateRunes().ToList();
        if (scalars.Count == 0)
        {
            throw NormalizeException.Of(NormalizeErrorKind.Internal);
        }
        if (scalars.Count == 1 && Spelling.SymbolName(scalars[0].Value) is { } name)
        {
            output.Word(name);
            return false;
        }
        // A multi-scalar cluster is owned as a whole, with every scalar retained.
        foreach (var scalar in scalars)
        {
            CodePoint(scalar, output);
        }
        return true;
    }

    static void CodePoint(Rune scalar, FallbackOutput output)
    {
        output.Word("unikod u artı");
        foreach (var hexadecimal in scalar.Value.ToString("X4", CultureInfo.InvariantCulture))
        {
            var name = Numerals.DigitName(hexadecimal) ?? Spelling.LetterName(hexadecimal)
                ?? throw NormalizeException.Of(NormalizeErrorKind.Internal);
            output.Word(name);
        }
    }
}
