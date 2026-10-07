// src/classify/context.rs portu
namespace NormalizerTr;

internal static class CueContext
{
    public static readonly string[] DateCues = { "tarih", "tarihi" };
    public static readonly string[] TimeCues = { "saat" };

    /// <summary>Casing and optional colon removal apply only to reviewed lookup keys.</summary>
    public static string CueKey(string text) => Lexicon.LookupKey(text.EndsWith(':') ? text[..^1] : text);

    public static bool IsCueWord(string text)
    {
        var key = CueKey(text);
        return DateCues.Contains(key) || TimeCues.Contains(key);
    }

    public static bool IsClockWord(string text) => !text.Contains(':') && TimeCues.Contains(CueKey(text));

    /// <summary>Gövdenin başladığı UTF-16 ofseti (kelime + ':').</summary>
    public static int? InlinePrefix(string text)
    {
        var colon = text.IndexOf(':');
        if (colon < 0)
        {
            return null;
        }
        var word = text[..colon];
        var body = text[(colon + 1)..];
        return IsCueWord(word) && body.Length > 0 && Uni.IsAsciiDigit(body[0]) ? word.Length + 1 : null;
    }
}
