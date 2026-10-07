// src/domain/electronic.rs portu
using System.Globalization;
using System.Text;

namespace NormalizerTr;

internal sealed class Electronic
{
    readonly string spoken;

    Electronic(string spoken)
    {
        this.spoken = spoken;
    }

    public string Render() => spoken;

    static string? Tld(string text) => Uni.AsciiLower(text) switch
    {
        "com" => "kom",
        "net" => "net",
        "org" => "org",
        "tr" => "te re",
        "gov" => "gov",
        "edu" => "edu",
        "app" => "app",
        _ => null,
    };

    static bool Host(string text)
    {
        if (text.Length > 253 || text.Length == 0 || !Uni.IsAscii(text))
        {
            return false;
        }
        var labels = text.Split('.');
        return labels.Length >= 2
               && labels.All(label =>
                   label.Length > 0
                   && label.Length <= 63
                   && !label.StartsWith('-')
                   && !label.EndsWith('-')
                   && !(label.Length >= 4 && Uni.EqIgnoreAsciiCase(label[..4], "xn--"))
                   && Uni.AllChars(label, b => Uni.IsAsciiAlnum(b) || b == '-'))
               && Tld(labels[^1]) != null;
    }

    static string Characters(string text, bool domain)
    {
        var words = new List<string>();
        var offset = 0;
        while (offset < text.Length)
        {
            var b = text[offset];
            if (Uni.IsAsciiAlpha(b))
            {
                var start = offset;
                while (offset < text.Length && Uni.IsAsciiAlpha(text[offset]))
                {
                    offset++;
                }
                var chunk = text[start..offset];
                words.Add(domain ? Tld(chunk) ?? chunk : chunk);
            }
            else if (Uni.IsAsciiDigit(b))
            {
                var start = offset;
                while (offset < text.Length && Uni.IsAsciiDigit(text[offset]))
                {
                    offset++;
                }
                words.Add(Numerals.DigitsText(text[start..offset]));
            }
            else
            {
                var word = b switch
                {
                    '.' => "nokta",
                    '@' => "et",
                    '/' => "eğik çizgi",
                    ':' => "iki nokta",
                    '?' => "soru işareti",
                    '=' => "eşittir",
                    '&' => "ve",
                    '#' => "kare",
                    '%' => "yüzde",
                    '-' => "tire",
                    '_' => "alt çizgi",
                    '+' => "artı",
                    '(' => "aç parantez",
                    ')' => "kapat parantez",
                    '!' => "ünlem",
                    '$' => "dolar işareti",
                    '\'' => "kesme",
                    '*' => "yıldız",
                    ',' => "virgül",
                    ';' => "noktalı virgül",
                    '~' => "tilde",
                    _ => "",
                };
                if (word.Length > 0)
                {
                    words.Add(word);
                }
                offset++;
            }
        }
        return string.Join(" ", words);
    }

    public static Electronic? Parse(string text, bool bare)
    {
        if (!Uni.IsAscii(text) || text.Any(char.IsControl))
        {
            return null;
        }
        if (text.Contains('@') && !text.Contains("://", StringComparison.Ordinal) && !text.StartsOrd("www."))
        {
            if (text.Contains("://", StringComparison.Ordinal) || Uni.Count(text, '@') != 1)
            {
                return null;
            }
            var at = text.IndexOf('@');
            var local = text[..at];
            var domain = text[(at + 1)..];
            if (local.Length == 0
                || local.Length > 64
                || local.StartsWith('.')
                || local.EndsWith('.')
                || local.Contains("..", StringComparison.Ordinal)
                || !Uni.AllChars(local, b => Uni.IsAsciiAlnum(b) || b is '.' or '_' or '+' or '-')
                || !Host(domain))
            {
                return null;
            }
            return new Electronic($"{Characters(local, false)} et {DomainWords(domain)}");
        }
        string prefix;
        string body;
        if (text.Length >= 8 && Uni.EqIgnoreAsciiCase(text[..8], "https://"))
        {
            prefix = "ha te te pe es iki nokta eğik çizgi eğik çizgi ";
            body = text[8..];
        }
        else if (text.Length >= 7 && Uni.EqIgnoreAsciiCase(text[..7], "http://"))
        {
            prefix = "ha te te pe iki nokta eğik çizgi eğik çizgi ";
            body = text[7..];
        }
        else if (text.StartsOrd("www.") || bare)
        {
            prefix = "";
            body = text;
        }
        else
        {
            return null;
        }
        if (body.Length == 0)
        {
            return null;
        }
        var split = body.IndexOfAny(new[] { '/', '?', '#' });
        if (split < 0)
        {
            split = body.Length;
        }
        var authority = body[..split];
        var rest = body[split..];
        string hostPart;
        string? port = null;
        var colon = authority.IndexOf(':');
        if (colon >= 0)
        {
            hostPart = authority[..colon];
            port = authority[(colon + 1)..];
            if (port.Length == 0 || port.Length > 5 || !Uni.AllAsciiDigits(port))
            {
                return null;
            }
            var portNumber = uint.Parse(port, NumberStyles.None, CultureInfo.InvariantCulture);
            if (portNumber is < 1 or > 65535)
            {
                return null;
            }
        }
        else
        {
            hostPart = authority;
        }
        if (!Host(hostPart) || authority.Contains('@'))
        {
            return null;
        }
        var depth = 0;
        var i = 0;
        while (i < rest.Length)
        {
            var b = rest[i];
            if (b == '%')
            {
                if (i + 2 >= rest.Length || !Uni.IsAsciiHex(rest[i + 1]) || !Uni.IsAsciiHex(rest[i + 2]))
                {
                    return null;
                }
                i += 3;
                continue;
            }
            if (!(Uni.IsAsciiAlnum(b) || "-._~!$&'()*+,;=:@/?#".Contains(b)))
            {
                return null;
            }
            if (b == '(')
            {
                depth++;
            }
            if (b == ')')
            {
                depth--;
            }
            if (depth < 0)
            {
                return null;
            }
            i++;
        }
        if (depth != 0 || Uni.Count(rest, '#') > 1)
        {
            return null;
        }
        var spoken = new StringBuilder(prefix).Append(DomainWords(hostPart));
        if (port != null)
        {
            spoken.Append(" iki nokta ");
            spoken.Append(Numerals.DigitsText(port));
        }
        if (rest.Length > 0)
        {
            spoken.Append(' ');
            spoken.Append(Characters(rest, false));
        }
        return new Electronic(spoken.ToString());
    }

    static string DomainWords(string domain)
    {
        var parts = domain.Split('.');
        return string.Join(" nokta ", parts.Select((part, i) =>
            i == parts.Length - 1 ? Tld(part) ?? part
            : i == 0 && part == "www" ? "çift ve çift ve çift ve"
            : Characters(part, false)));
    }

    public static bool LooksLike(string text) =>
        text.Contains('@') || text.Contains("://", StringComparison.Ordinal) || text.StartsOrd("www.");

    public static string? Hashtag(string text)
    {
        if (!text.StartsWith('#'))
        {
            return null;
        }
        var body = text[1..];
        if (body.Length == 0
            || !Uni.All(body, c => Uni.IsAlphanumeric(c) || c.Value == '_')
            || Uni.Any(body, c => Uni.IsNumeric(c) && !Uni.IsAsciiDigit(c)))
        {
            return null;
        }
        var output = new StringBuilder("hashtag ");
        var chunk = new StringBuilder();
        foreach (var ch in body.EnumerateRunes())
        {
            if (Uni.IsAsciiDigit(ch))
            {
                if (chunk.Length > 0)
                {
                    output.Append(chunk).Append(' ');
                    chunk.Clear();
                }
                output.Append(Numerals.DigitsText(ch.ToString())).Append(' ');
            }
            else
            {
                chunk.Append(ch.ToString());
            }
        }
        output.Append(chunk);
        return output.ToString().TrimEnd();
    }
}
