// src/classify/boundaries.rs portu
namespace NormalizerTr;

internal readonly record struct Claim(Rng Range, int Next);

/// <summary>The one source of cursor advancement, source claims and whole-hint ownership.</summary>
internal sealed class Boundaries
{
    readonly List<Token> tokens;
    readonly List<Rng> phoneLike;

    public Boundaries(List<Token> tokens, List<Rng> phoneLike)
    {
        this.tokens = tokens;
        this.phoneLike = phoneLike;
    }

    public Claim Claim(int start, int end)
    {
        if (start < 0 || start >= tokens.Count || end < start || end >= tokens.Count)
        {
            throw NormalizeException.Of(NormalizeErrorKind.Internal);
        }
        return new Claim(new Rng(tokens[start].Range.Start, tokens[end].Range.End), end + 1);
    }

    public Claim HintClaim(InternalHint hint, Claim? detected)
    {
        var range = hint.Range;
        var start = Seq.PartitionPoint(tokens, t => t.Range.End <= range.Start);
        var end = Seq.PartitionPoint(tokens, t => t.Range.Start < range.End);
        var touched = tokens.GetRange(start, end - start);
        var wrapAllowed = hint.Kind is HintKind.Digits or HintKind.Telephone;
        var phoneStart = Seq.PartitionPoint(phoneLike, p => p.End <= range.Start);
        var phoneEnd = Seq.PartitionPoint(phoneLike, p => p.Start < range.End);
        var phones = phoneLike.GetRange(phoneStart, phoneEnd - phoneStart);
        if (touched.Count == 0
            || (!wrapAllowed && (touched[0].Range.Start != range.Start || touched[^1].Range.End != range.End))
            || touched.Any(t => t.Range.Start < range.Start || t.Range.End > range.End)
            || phones.Any(p => Scan.Overlaps(p, range) && (p != range || !wrapAllowed)))
        {
            throw NormalizeException.Of(NormalizeErrorKind.InvalidHint);
        }
        if (detected is { } span && (span.Range.Start < range.Start || span.Range.End > range.End))
        {
            throw NormalizeException.Of(NormalizeErrorKind.InvalidHint);
        }
        return new Claim(range, end);
    }

    public Rng? PhoneAt(int start)
    {
        var index = Seq.BinarySearch(phoneLike, start, p => p.Start);
        return index >= 0 ? phoneLike[index] : null;
    }

    public int NextAt(int end) => Seq.PartitionPoint(tokens, t => t.Range.Start < end);

    public static bool OverlapsHint(List<InternalHint> hints, Rng range)
    {
        var index = Seq.PartitionPoint(hints, h => h.Range.End <= range.Start);
        return index < hints.Count && Scan.Overlaps(hints[index].Range, range);
    }

    public static bool QuantityTail(string text)
    {
        if (text.StartsWith('\'') || text.StartsWith('’'))
        {
            return true;
        }
        var @base = Uni.Head(text);
        return NumericText.Label(@base)
               || NumericText.UnsupportedLabel(@base)
               || NumericText.UnsupportedLabel(Uni.AsciiUpper(@base))
               || (Uni.Len8(@base) == 3 && Uni.AllChars(@base, Uni.IsAsciiUpper) && Lexicon.Abbreviation(@base) == null)
               || Lexicon.UnitMarker(@base)
               || @base == "%";
    }
}
