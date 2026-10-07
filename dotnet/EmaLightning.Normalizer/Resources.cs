// src/resources.rs portu — derlenmiş sabit desenler
using System.Text.RegularExpressions;

namespace NormalizerTr;

internal static class Resources
{
    public static readonly Regex Tokens = new(@"\S+", RegexOptions.CultureInvariant);

    public static readonly Regex PhoneLike =
        new(@"(?:\+?[0-9]{1,3})(?:[ \t]+(?:\([0-9]{2,}\)|[0-9]{2,})){2,}", RegexOptions.CultureInvariant);

    public static readonly Regex QuotedEmail = new("\"[^\"\\r\\n]{0,128}\"@[^\\s]+", RegexOptions.CultureInvariant);
}
