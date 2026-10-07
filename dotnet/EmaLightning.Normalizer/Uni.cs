// Rust str/char davranışlarının C# karşılıkları.
// İç indeksler UTF-16'dır; Rust'taki bayt uzunluğu karşılaştırmaları Len8 ile birebir korunur.
using System.Globalization;
using System.Text;

namespace NormalizerTr;

/// <summary>Tanıma metni içindeki iç aralık (UTF-16 indeksleri).</summary>
internal readonly record struct Rng(int Start, int End);

/// <summary>Rust Result karşılığı.</summary>
internal readonly struct Res<T, E>
{
    Res(bool ok, T value, E error)
    {
        IsOk = ok;
        Value = value;
        Error = error;
    }

    public bool IsOk { get; }
    public T Value { get; }
    public E Error { get; }

    public static Res<T, E> Ok(T value) => new(true, value, default!);
    public static Res<T, E> Err(E error) => new(false, default!, error);
}

internal static class Uni
{
    public static int Len8(string s) => Encoding.UTF8.GetByteCount(s);

    // char::is_alphabetic — Alphabetic özelliği: harf kategorileri + Nl.
    // Not: Unicode Other_Alphabetic (bazı birleşik işaretler) .NET'te hazır yok; dahil değil.
    public static bool IsAlphabetic(Rune r) => Rune.GetUnicodeCategory(r) switch
    {
        UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter
            or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter or UnicodeCategory.LetterNumber => true,
        _ => false,
    };

    // char::is_numeric — Nd, Nl, No
    public static bool IsNumeric(Rune r) => Rune.GetUnicodeCategory(r) switch
    {
        UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber => true,
        _ => false,
    };

    public static bool IsAlphanumeric(Rune r) => IsAlphabetic(r) || IsNumeric(r);

    // char::is_uppercase — Lu + Other_Uppercase
    public static bool IsUppercase(Rune r) =>
        Rune.GetUnicodeCategory(r) == UnicodeCategory.UppercaseLetter || OtherUppercase(r.Value);

    static bool OtherUppercase(int v) =>
        v is (>= 0x2160 and <= 0x216F) or (>= 0x24B6 and <= 0x24CF) or (>= 0x1F130 and <= 0x1F149)
            or (>= 0x1F150 and <= 0x1F169) or (>= 0x1F170 and <= 0x1F189);

    public static bool IsWhitespace(Rune r) => Rune.IsWhiteSpace(r);

    // unicode_normalization::char::is_combining_mark — Mn, Mc, Me
    public static bool IsCombiningMark(Rune r) => Rune.GetUnicodeCategory(r) switch
    {
        UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark => true,
        _ => false,
    };

    public static bool IsAsciiDigit(char c) => c is >= '0' and <= '9';
    public static bool IsAsciiDigit(Rune r) => r.Value is >= '0' and <= '9';
    public static bool IsAsciiAlpha(char c) => c is (>= 'a' and <= 'z') or (>= 'A' and <= 'Z');
    public static bool IsAsciiAlnum(char c) => IsAsciiAlpha(c) || IsAsciiDigit(c);
    public static bool IsAsciiUpper(char c) => c is >= 'A' and <= 'Z';
    public static bool IsAsciiHex(char c) => IsAsciiDigit(c) || c is (>= 'a' and <= 'f') or (>= 'A' and <= 'F');

    public static bool IsAscii(string s)
    {
        foreach (var c in s)
        {
            if (c > 0x7F)
            {
                return false;
            }
        }
        return true;
    }

    // Rust: s.bytes().all(|b| b.is_ascii_digit()) — boş dizgede true
    public static bool AllAsciiDigits(string s) => AllChars(s, IsAsciiDigit);
    public static bool AnyAsciiDigit(string s) => AnyChars(s, IsAsciiDigit);

    public static bool AllChars(string s, Func<char, bool> predicate)
    {
        foreach (var c in s)
        {
            if (!predicate(c))
            {
                return false;
            }
        }
        return true;
    }

    public static bool AnyChars(string s, Func<char, bool> predicate)
    {
        foreach (var c in s)
        {
            if (predicate(c))
            {
                return true;
            }
        }
        return false;
    }

    public static bool All(string s, Func<Rune, bool> predicate)
    {
        foreach (var r in s.EnumerateRunes())
        {
            if (!predicate(r))
            {
                return false;
            }
        }
        return true;
    }

    public static bool Any(string s, Func<Rune, bool> predicate)
    {
        foreach (var r in s.EnumerateRunes())
        {
            if (predicate(r))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Rust str::find(closure): koşulu sağlayan ilk rune'un UTF-16 indeksi, yoksa uzunluk.</summary>
    public static int Find(string s, Func<Rune, bool> predicate)
    {
        var i = 0;
        while (i < s.Length)
        {
            Rune.DecodeFromUtf16(s.AsSpan(i), out var r, out var n);
            if (predicate(r))
            {
                return i;
            }
            i += n;
        }
        return s.Length;
    }

    public static Rune? RuneBefore(string s, int end)
    {
        if (end <= 0)
        {
            return null;
        }
        Rune.DecodeLastFromUtf16(s.AsSpan(0, end), out var r, out _);
        return r;
    }

    public static Rune? RuneAt(string s, int start)
    {
        if (start >= s.Length)
        {
            return null;
        }
        Rune.DecodeFromUtf16(s.AsSpan(start), out var r, out _);
        return r;
    }

    public static Rune FirstRune(string s)
    {
        Rune.DecodeFromUtf16(s.AsSpan(), out var r, out _);
        return r;
    }

    public static int GraphemeLength(ReadOnlySpan<char> s) => StringInfo.GetNextTextElementLength(s);

    /// <summary>text.split(['\'', '’']).next() — her zaman ilk parça.</summary>
    public static string Head(string s)
    {
        var i = s.IndexOfAny(Apostrophes);
        return i < 0 ? s : s[..i];
    }

    public static readonly char[] Apostrophes = { '\'', '’' };

    public static string AsciiUpper(string s)
    {
        var chars = s.ToCharArray();
        for (var i = 0; i < chars.Length; i++)
        {
            if (chars[i] is >= 'a' and <= 'z')
            {
                chars[i] = (char)(chars[i] - 32);
            }
        }
        return new string(chars);
    }

    public static string AsciiLower(string s)
    {
        var chars = s.ToCharArray();
        for (var i = 0; i < chars.Length; i++)
        {
            if (chars[i] is >= 'A' and <= 'Z')
            {
                chars[i] = (char)(chars[i] + 32);
            }
        }
        return new string(chars);
    }

    public static bool EqIgnoreAsciiCase(string a, string b) => a.Length == b.Length && AsciiLower(a) == AsciiLower(b);

    public static int Count(string s, char c)
    {
        var n = 0;
        foreach (var x in s)
        {
            if (x == c)
            {
                n++;
            }
        }
        return n;
    }

    public static bool StartsOrd(this string s, string prefix) => s.StartsWith(prefix, StringComparison.Ordinal);
    public static bool EndsOrd(this string s, string suffix) => s.EndsWith(suffix, StringComparison.Ordinal);

    public static string TrimStartBy(string s, Func<char, bool> predicate)
    {
        var i = 0;
        while (i < s.Length && predicate(s[i]))
        {
            i++;
        }
        return s[i..];
    }

    public static string TrimEndBy(string s, Func<char, bool> predicate)
    {
        var i = s.Length;
        while (i > 0 && predicate(s[i - 1]))
        {
            i--;
        }
        return s[..i];
    }

    /// <summary>Rust split_once(char seçenekleri): ilk ayırıcıda iki parça.</summary>
    public static bool SplitOnce(string s, char[] separators, out string left, out string right)
    {
        var i = s.IndexOfAny(separators);
        if (i < 0)
        {
            left = right = "";
            return false;
        }
        left = s[..i];
        right = s[(i + 1)..];
        return true;
    }
}

/// <summary>Sıralı listede Rust partition_point / binary_search karşılıkları.</summary>
internal static class Seq
{
    public static int PartitionPoint<T>(IReadOnlyList<T> list, Func<T, bool> predicate)
    {
        int lo = 0, hi = list.Count;
        while (lo < hi)
        {
            var mid = (lo + hi) / 2;
            if (predicate(list[mid]))
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        return lo;
    }

    public static int BinarySearch<T>(IReadOnlyList<T> list, int key, Func<T, int> selector)
    {
        int lo = 0, hi = list.Count - 1;
        while (lo <= hi)
        {
            var mid = (lo + hi) / 2;
            var value = selector(list[mid]);
            if (value == key)
            {
                return mid;
            }
            if (value < key)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }
        return -1;
    }
}
