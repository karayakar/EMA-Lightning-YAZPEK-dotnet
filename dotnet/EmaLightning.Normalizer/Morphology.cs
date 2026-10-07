// src/morphology.rs portu
namespace NormalizerTr;

internal enum Inflection
{
    Ordinal,
    Accusative,
    Dative,
    Locative,
    Ablative,
    Genitive,
    Derivation,
}

internal enum Harmony
{
    BackFlat,
    FrontFlat,
    BackRound,
    FrontRound,
}

internal enum WordEnd
{
    Vowel,
    Voiced,
    Voiceless,
    Softens,
    SoftensP,
    Possessive,
}

internal readonly record struct Word(string Text, Harmony Harmony, WordEnd End)
{
    static string High(Harmony h) => h switch
    {
        Harmony.BackFlat => "ı",
        Harmony.FrontFlat => "i",
        Harmony.BackRound => "u",
        _ => "ü",
    };

    static string Low(Harmony h) => h is Harmony.BackFlat or Harmony.BackRound ? "a" : "e";

    public string SourceSuffix(Inflection inflection)
    {
        var high = High(Harmony);
        var low = Low(Harmony);
        var vowel = End is WordEnd.Vowel or WordEnd.Possessive;
        var stop = End is WordEnd.Voiceless or WordEnd.Softens or WordEnd.SoftensP ? "t" : "d";
        var possessive = End == WordEnd.Possessive;
        return inflection switch
        {
            Inflection.Accusative when possessive => $"n{high}",
            Inflection.Dative when possessive => $"n{low}",
            Inflection.Locative when possessive => $"nd{low}",
            Inflection.Ablative when possessive => $"nd{low}n",
            Inflection.Ordinal when vowel => $"nc{high}",
            Inflection.Ordinal => $"{high}nc{high}",
            Inflection.Accusative when vowel => $"y{high}",
            Inflection.Accusative => high,
            Inflection.Dative when vowel => $"y{low}",
            Inflection.Dative => low,
            Inflection.Locative => $"{stop}{low}",
            Inflection.Ablative => $"{stop}{low}n",
            Inflection.Genitive when vowel => $"n{high}n",
            Inflection.Genitive => $"{high}n",
            _ => $"l{high}k",
        };
    }
}

/// <summary>Carries final spoken-word metadata; never recovers morphology by parsing text.</summary>
internal sealed class Spoken
{
    string text;
    Word tail;

    Spoken(string text, Word tail)
    {
        this.text = text;
        this.tail = tail;
    }

    public static Spoken Lexical(string text, Word tail) => new(text, tail);

    public static Spoken FromWords(IEnumerable<Word> words)
    {
        var spoken = new Spoken("", new Word("sıfır", Harmony.BackFlat, WordEnd.Voiced));
        foreach (var word in words)
        {
            spoken.AppendWord(word);
        }
        return spoken;
    }

    public void Prefix(string prefix) => text = prefix + text;

    public void AppendLiteral(string literal) => text += literal;

    public void AppendWord(Word word)
    {
        if (text.Length > 0)
        {
            text += " ";
        }
        text += word.Text;
        tail = word;
    }

    public string SourceSuffix(Inflection inflection) => tail.SourceSuffix(inflection);

    public void Inflect(Inflection inflection)
    {
        var suffix = SourceSuffix(inflection);
        var vowelSuffix = inflection is Inflection.Ordinal or Inflection.Accusative or Inflection.Dative
            or Inflection.Genitive;
        if (tail.End == WordEnd.Softens && vowelSuffix)
        {
            text = text[..^1] + "d";
        }
        if (tail.End == WordEnd.SoftensP && vowelSuffix)
        {
            text = text[..^1] + "b";
        }
        text += suffix;
        if (inflection == Inflection.Ordinal)
        {
            tail = tail with { End = WordEnd.Vowel };
        }
        else if (inflection == Inflection.Derivation)
        {
            tail = tail with { End = WordEnd.Voiceless };
        }
    }

    public string IntoText() => text;
}

internal static class Morphology
{
    static readonly Inflection[] IntegerFamilies =
    {
        Inflection.Ordinal, Inflection.Accusative, Inflection.Dative, Inflection.Locative, Inflection.Ablative,
        Inflection.Genitive,
    };

    static readonly Inflection[] CaseFamilies =
    {
        Inflection.Accusative, Inflection.Dative, Inflection.Locative, Inflection.Ablative, Inflection.Genitive,
    };

    public static Inflection? IntegerInflection(Spoken spoken, string suffix)
    {
        foreach (var inflection in IntegerFamilies)
        {
            if (spoken.SourceSuffix(inflection) == suffix)
            {
                return inflection;
            }
        }
        return null;
    }

    public static Inflection? CaseInflection(Word source, string suffix)
    {
        foreach (var family in CaseFamilies)
        {
            if (source.SourceSuffix(family) == suffix)
            {
                return family;
            }
        }
        return null;
    }

    public static Inflection? SpokenCase(Spoken spoken, string suffix)
    {
        foreach (var family in CaseFamilies)
        {
            if (spoken.SourceSuffix(family) == suffix)
            {
                return family;
            }
        }
        return null;
    }
}
