// src/domain/identifiers.rs portu
using System.Text;

namespace NormalizerTr;

internal sealed class Telephone
{
    readonly string national;
    readonly bool international;
    readonly bool nationalZero;

    Telephone(string national, bool international, bool nationalZero)
    {
        this.national = national;
        this.international = international;
        this.nationalZero = nationalZero;
    }

    static readonly char[] GroupSeparators = { ' ', '-', '(', ')' };

    public static Telephone? Parse(string text, bool @explicit)
    {
        if (!Uni.AllChars(text, c => Uni.IsAsciiDigit(c) || c is '+' or ' ' or '-' or '(' or ')'))
        {
            return null;
        }
        if (Uni.Count(text, '+') > (text.StartsWith('+') ? 1 : 0))
        {
            return null;
        }
        var depth = 0;
        foreach (var ch in text)
        {
            if (ch == '(' && depth == 0)
            {
                depth = 1;
            }
            else if (ch == ')' && depth == 1)
            {
                depth = 0;
            }
            else if (ch is '(' or ')')
            {
                return null;
            }
        }
        if (depth != 0
            || text.EndsWith('-') || text.EndsWith(' ') || text.EndsWith('(')
            || text.Contains("--", StringComparison.Ordinal)
            || text.Contains("  ", StringComparison.Ordinal))
        {
            return null;
        }
        var international = text.StartsOrd("+90");
        var cleaned = new string(text.Where(Uni.IsAsciiDigit).ToArray());
        string national;
        bool nationalZero;
        if (international)
        {
            if (cleaned.Length != 12)
            {
                return null;
            }
            national = cleaned[2..];
            nationalZero = false;
        }
        else if (cleaned.Length == 11 && cleaned.StartsWith('0'))
        {
            national = cleaned[1..];
            nationalZero = true;
        }
        else if (cleaned.Length == 10 && @explicit)
        {
            national = cleaned;
            nationalZero = false;
        }
        else
        {
            return null;
        }
        var groups = text.Split(GroupSeparators).Where(s => s.Length > 0).ToList();
        if (groups.Count > 1)
        {
            var sizes = groups.Select(s => Uni.Len8(s.TrimStart('+'))).ToArray();
            bool allowed;
            if (international)
            {
                allowed = sizes.SequenceEqual(new[] { 2, 3, 3, 2, 2 });
            }
            else if (nationalZero)
            {
                allowed = sizes.SequenceEqual(new[] { 4, 3, 2, 2 }) || sizes.SequenceEqual(new[] { 1, 3, 3, 2, 2 });
            }
            else
            {
                allowed = @explicit && sizes.SequenceEqual(new[] { 3, 3, 2, 2 });
            }
            if (!allowed)
            {
                return null;
            }
        }
        return new Telephone(national, international, nationalZero);
    }

    public string Render()
    {
        var chunks = new List<string>();
        if (international)
        {
            chunks.Add("artı doksan");
        }
        if (nationalZero)
        {
            chunks.Add("sıfır");
        }
        foreach (var (start, end) in new[] { (0, 3), (3, 6), (6, 8), (8, 10) })
        {
            var digits = national[start..end];
            string output;
            if (digits.StartsWith('0'))
            {
                output = Numerals.DigitsText(digits);
            }
            else
            {
                // Kurucu her grubu en çok üç ASCII rakamla sınırlar.
                ulong value = 0;
                foreach (var c in digits)
                {
                    value = value * 10 + (ulong)(c - '0');
                }
                output = Numerals.Cardinal(value).IntoText();
            }
            chunks.Add(output);
        }
        return string.Join(" ", chunks);
    }
}

internal sealed class Iban
{
    readonly string canonical;

    Iban(string canonical)
    {
        this.canonical = canonical;
    }

    public static Iban? Parse(string text)
    {
        var groups = text.Split(' ');
        if (groups.Any(g => g.Length == 0))
        {
            return null;
        }
        if (groups.Length > 1
            && (groups.Length != 7 || groups.Take(6).Any(g => Uni.Len8(g) != 4) || Uni.Len8(groups[6]) != 2))
        {
            return null;
        }
        var canonical = string.Concat(groups);
        if (Uni.Len8(canonical) != 26 || !canonical.StartsOrd("TR") || !Uni.AllAsciiDigits(canonical[2..]))
        {
            return null;
        }
        var check = (canonical[2] - '0') * 10 + (canonical[3] - '0');
        if (check is < 2 or > 98)
        {
            return null;
        }
        var remainder = 0u;
        foreach (var c in canonical[4..] + "2927" + canonical[2..4])
        {
            remainder = (remainder * 10 + (uint)(c - '0')) % 97;
        }
        return remainder == 1 ? new Iban(canonical) : null;
    }

    public string Render()
    {
        var output = new StringBuilder("te re ");
        output.Append(Numerals.DigitsText(canonical[2..4]));
        var rest = canonical[4..];
        for (var i = 0; i < rest.Length; i += 4)
        {
            output.Append(", ");
            output.Append(Numerals.DigitsText(rest.Substring(i, Math.Min(4, rest.Length - i))));
        }
        return output.ToString();
    }
}
