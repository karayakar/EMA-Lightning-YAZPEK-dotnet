// frontend.py portu: yazılı Türkçe → modelin alfabesindeki metin. Metin yüzünden asla hata fırlatmaz.
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using NormalizerTr;

namespace EmaLightning;

internal sealed class Frontend
{
    const int BlockBytes = 8 * 1024;
    const string Turkish = "çğıöşüÇĞİÖŞÜ";

    static readonly Dictionary<char, string> Typography = new()
    {
        ['’'] = "'", ['‘'] = "'", ['ʼ'] = "'", ['´'] = "'", ['`'] = "'", ['“'] = "\"", ['”'] = "\"", ['„'] = "\"",
        ['«'] = "\"", ['»'] = "\"", ['–'] = "-", ['—'] = "-", ['−'] = "-", ['…'] = "...",
    };

    static readonly Regex Unsafe =
        new("[\u0000-\u0008\u000b-\u001f\u007f-\u009f؜‎‏‪-‮⁦-⁩]");

    static readonly Regex Spaces = new(@"\s+");

    readonly HashSet<string> vocab;
    readonly Normalizer normalizer = new();

    public Frontend(IEnumerable<string> vocab)
    {
        this.vocab = new HashSet<string>(vocab, StringComparer.Ordinal);
    }

    public string Process(string text)
    {
        // Python: text.encode("utf-8", "ignore").decode() — eşsiz vekiller düşer
        text = Unsafe.Replace(DropLoneSurrogates(text), " ");
        if (text.Trim().Length == 0)
        {
            return "";
        }
        return Alphabet(string.Join(" ", Blocks(text).Select(Spoken)));
    }

    string Spoken(string text)
    {
        try
        {
            return normalizer.Normalize(text, new NormalizeOptions { AmbiguityPolicy = AmbiguityPolicy.Fallback })
                .NormalizedText;
        }
        catch (NormalizeException)
        {
            // invalid input or a resource limit: keep the words rather than lose the sentence
            return text;
        }
    }

    string Alphabet(string text)
    {
        var mapped = new StringBuilder(text.Length);
        foreach (var c in text)
        {
            if (Typography.TryGetValue(c, out var replacement))
            {
                mapped.Append(replacement);
            }
            else
            {
                mapped.Append(c);
            }
        }
        var lowered = mapped.ToString().Replace("İ", "i").Replace("I", "ı");
        var output = new StringBuilder(lowered.Length);
        foreach (var rune in lowered.EnumerateRunes())
        {
            // Python str.lower() karşılığı (rune bazında)
            var ch = Rune.ToLowerInvariant(rune).ToString();
            if (!Turkish.Contains(ch, StringComparison.Ordinal))
            {
                // NFKD ve birleşik işaretleri at (Python: unicodedata.combining(c) != 0)
                var decomposed = new StringBuilder();
                foreach (var part in ch.Normalize(NormalizationForm.FormKD).EnumerateRunes())
                {
                    if (!IsCombining(part))
                    {
                        decomposed.Append(part.ToString());
                    }
                }
                ch = decomposed.ToString();
            }
            output.Append(ch.Length > 0 && ch.EnumerateRunes().All(r => vocab.Contains(r.ToString())) ? ch : " ");
        }
        return Spaces.Replace(output.ToString(), " ").Trim();
    }

    static bool IsCombining(Rune r) => Rune.GetUnicodeCategory(r) is UnicodeCategory.NonSpacingMark
        or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark;

    /// <summary>Split at whitespace into pieces the normalizer accepts in one call.</summary>
    static IEnumerable<string> Blocks(string text)
    {
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var block = new List<string>();
        var size = 0;
        foreach (var word in words)
        {
            var n = Encoding.UTF8.GetByteCount(word) + 1;
            if (block.Count > 0 && size + n > BlockBytes)
            {
                yield return string.Join(" ", block);
                block.Clear();
                size = 0;
            }
            block.Add(word);
            size += n;
        }
        if (block.Count > 0)
        {
            yield return string.Join(" ", block);
        }
    }

    static string DropLoneSurrogates(string text)
    {
        StringBuilder? sb = null;
        for (var i = 0; i < text.Length; i++)
        {
            var c = text[i];
            var valid = !char.IsSurrogate(c)
                        || (char.IsHighSurrogate(c) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]));
            if (char.IsHighSurrogate(c) && valid)
            {
                sb?.Append(c).Append(text[i + 1]);
                i++;
                continue;
            }
            if (!valid)
            {
                sb ??= new StringBuilder(text, 0, i, text.Length);
                continue;
            }
            sb?.Append(c);
        }
        return sb?.ToString() ?? text;
    }
}
