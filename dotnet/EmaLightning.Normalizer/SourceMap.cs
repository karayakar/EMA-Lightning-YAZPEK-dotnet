// src/source_map.rs portu
// Tanıma metni NFC'dir; her grafem sınırı için tanıma (UTF-16), özgün (UTF-16) ve özgün (UTF-8 bayt) ofsetleri tutulur.
using System.Text;

namespace NormalizerTr;

internal readonly record struct InternalHint(Rng Range, HintKind Kind);

internal sealed class SourceMap
{
    readonly string input;
    readonly int[] rec;
    readonly int[] orig;
    readonly int[] orig8;
    readonly int[] u8;

    public SourceMap(string input, WorkControl control)
    {
        control.Check();
        this.input = input;
        u8 = Utf8Offsets(input);
        var nfc = input.IsNormalized(NormalizationForm.FormC);
        var recognition = nfc ? null : new StringBuilder(input.Length);
        var recBounds = new List<int> { 0 };
        var origBounds = new List<int> { 0 };
        var offset = 0;
        while (offset < input.Length)
        {
            control.Check();
            var length = Uni.GraphemeLength(input.AsSpan(offset));
            if (recognition == null)
            {
                recBounds.Add(offset + length);
            }
            else
            {
                recognition.Append(input.Substring(offset, length).Normalize(NormalizationForm.FormC));
                recBounds.Add(recognition.Length);
            }
            origBounds.Add(offset + length);
            offset += length;
        }
        Text = recognition?.ToString() ?? input;
        rec = recBounds.ToArray();
        orig = origBounds.ToArray();
        orig8 = orig.Select(o => u8[o]).ToArray();
    }

    /// <summary>NFC tanıma metni.</summary>
    public string Text { get; }

    /// <summary>Özgün metindeki UTF-16 indeksinin UTF-8 bayt ofseti.</summary>
    public int Utf8At(int originalIndex) => u8[originalIndex];

    public Rng Cover(Rng range)
    {
        var start = Seq.PartitionPoint(rec, b => b <= range.Start) - 1;
        var end = Seq.PartitionPoint(rec, b => b < range.End);
        return new Rng(rec[start], rec[end]);
    }

    /// <summary>Tanıma aralığını özgün UTF-16 aralığına çevirir; sınır değilse InvalidHint (Rust ile aynı).</summary>
    public Rng Original(Rng range) => new(orig[Lookup(range.Start)], orig[Lookup(range.End)]);

    int Lookup(int recognitionOffset)
    {
        var index = Array.BinarySearch(rec, recognitionOffset);
        if (index < 0)
        {
            throw NormalizeException.Of(NormalizeErrorKind.InvalidHint);
        }
        return index;
    }

    /// <summary>Özgün sınırdan önceki grafem (Rust: input[..start].graphemes().next_back()).</summary>
    public string? GraphemeBefore(int originalIndex)
    {
        var index = Array.BinarySearch(orig, originalIndex);
        if (index < 0)
        {
            throw NormalizeException.Of(NormalizeErrorKind.Internal);
        }
        return index == 0 ? null : input[orig[index - 1]..orig[index]];
    }

    public List<InternalHint> Hints(IReadOnlyList<Hint> hints)
    {
        var mapped = new List<InternalHint>(hints.Count);
        foreach (var hint in hints)
        {
            if (hint.Range.Start >= hint.Range.End)
            {
                throw NormalizeException.Of(NormalizeErrorKind.InvalidHint);
            }
            var start = Array.BinarySearch(orig8, hint.Range.Start);
            var end = Array.BinarySearch(orig8, hint.Range.End);
            if (start < 0 || end < 0)
            {
                throw NormalizeException.Of(NormalizeErrorKind.InvalidHint);
            }
            mapped.Add(new InternalHint(new Rng(rec[start], rec[end]), hint.Kind));
        }
        // Rust sort_by_key kararlıdır; OrderBy da kararlı
        mapped = mapped.OrderBy(h => h.Range.Start).ToList();
        for (var i = 0; i + 1 < mapped.Count; i++)
        {
            if (mapped[i].Range.End > mapped[i + 1].Range.Start)
            {
                throw NormalizeException.Of(NormalizeErrorKind.InvalidHint);
            }
        }
        return mapped;
    }

    static int[] Utf8Offsets(string s)
    {
        var offsets = new int[s.Length + 1];
        var bytes = 0;
        for (var i = 0; i < s.Length; i++)
        {
            offsets[i] = bytes;
            var c = s[i];
            if (c < 0x80)
            {
                bytes += 1;
            }
            else if (c < 0x800)
            {
                bytes += 2;
            }
            else if (char.IsHighSurrogate(c))
            {
                bytes += 4;
            }
            else if (!char.IsLowSurrogate(c))
            {
                bytes += 3;
            }
        }
        offsets[s.Length] = bytes;
        return offsets;
    }
}
