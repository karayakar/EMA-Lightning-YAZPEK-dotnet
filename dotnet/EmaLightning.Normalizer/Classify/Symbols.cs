// src/classify/symbols.rs portu
namespace NormalizerTr;

internal static class Symbols
{
    /// <summary>Supplement only unclaimed source; primary whole-expression ownership wins.</summary>
    public static void Supplement(SourceMap source, List<Candidate> candidates, WorkControl control)
    {
        var added = new List<Candidate>();
        var start = 0;
        var length = source.Text.Length;
        var ranges = candidates.Select(c => c.Range).Append(new Rng(length, length)).ToList();
        foreach (var end in ranges)
        {
            ScanGap(source, start, end.Start, added, candidates.Count, control);
            start = end.End;
        }
        candidates.AddRange(added);
        var sorted = candidates.OrderBy(c => c.Range.Start).ToList();
        candidates.Clear();
        candidates.AddRange(sorted);
    }

    static void ScanGap(SourceMap source, int start, int end, List<Candidate> added, int primaryCount,
        WorkControl control)
    {
        var text = source.Text;
        var cursor = start;
        int? open = null;
        while (cursor < end)
        {
            control.Check();
            var remaining = text.AsSpan(cursor, end - cursor);
            var graphemeLength = Uni.GraphemeLength(remaining);
            if (graphemeLength == 0)
            {
                throw NormalizeException.Of(NormalizeErrorKind.Internal);
            }
            var grapheme = remaining[..graphemeLength].ToString();
            var emoticon = FallbackRequest.EmoticonLength(remaining.ToString());
            var length = emoticon is { } e ? source.Cover(new Rng(cursor, cursor + e)).End - cursor : graphemeLength;
            if (length > end - cursor)
            {
                throw NormalizeException.Of(NormalizeErrorKind.Internal);
            }
            var withinWord = (Uni.RuneBefore(text, cursor) is { } before && Uni.IsAlphabetic(before))
                             || (Uni.RuneAt(text, cursor + graphemeLength) is { } after && Uni.IsAlphabetic(after));
            if (emoticon != null || FallbackRequest.NeedsReading(grapheme, withinWord))
            {
                open ??= cursor;
            }
            else if (open is { } begin)
            {
                Append(added, new Rng(begin, cursor), primaryCount);
                open = null;
            }
            cursor += length;
        }
        if (open is { } last)
        {
            Append(added, new Rng(last, end), primaryCount);
        }
    }

    static void Append(List<Candidate> added, Rng range, int primaryCount)
    {
        if (primaryCount + added.Count >= NormalizerLimits.MaxCandidates)
        {
            throw NormalizeException.Limited(LimitKind.Candidates);
        }
        added.Add(new Candidate(range, Reading.Of(FallbackRequest.Symbols())));
    }
}
