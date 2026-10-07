// src/pipeline.rs portu
namespace NormalizerTr;

internal static class Pipeline
{
    // Rust size_of::<T>() değerleri (64-bit, hizalamalı): yalnızca 512 KiB sonuç bütçesinin sınırını etkiler.
    const int ResultSize = 136;
    const int SegmentSize = 64;
    const int IssueSize = 40;
    const int DiagnosticSize = 24;

    sealed class ResultBudget
    {
        long used;

        public void Charge(long bytes)
        {
            if (bytes < 0 || used + bytes > NormalizerLimits.MaxResultBytes)
            {
                throw NormalizeException.Limited(LimitKind.Result);
            }
            used += bytes;
        }

        // Text is owned once per segment and once again by normalized_text.
        public void Segment(int textBytes) => Charge((long)textBytes * 2 + SegmentSize);

        public void Issue() => Charge(IssueSize);

        public int TextAllowance()
        {
            var rest = NormalizerLimits.MaxResultBytes - used - SegmentSize;
            if (rest < 0)
            {
                throw NormalizeException.Limited(LimitKind.Result);
            }
            return (int)(rest / 2);
        }
    }

    static string Explanation(IssueCategory category) => category switch
    {
        IssueCategory.Ambiguous => "expression requires an explicit supported cue or hint",
        IssueCategory.InvalidExpression => "expression has an invalid value, grammar, or suffix",
        IssueCategory.ProtectedIdentifier => "structured identifier is preserved in full",
        IssueCategory.Unsupported => "expression is outside the bounded profile",
        _ => "uppercase abbreviation is not approved",
    };

    static bool Forbidden(string input)
    {
        for (var i = 0; i < input.Length; i++)
        {
            var c = input[i];
            // Rust dizgileri eşsiz vekil (lone surrogate) taşıyamaz; Python bağlayıcısı da invalid_input verir.
            if (char.IsHighSurrogate(c))
            {
                if (i + 1 >= input.Length || !char.IsLowSurrogate(input[i + 1]))
                {
                    return true;
                }
                i++;
                continue;
            }
            if (char.IsLowSurrogate(c))
            {
                return true;
            }
            if ((char.IsControl(c) && c is not ('\n' or '\r' or '\t'))
                || c is '؜' or '‎' or '‏' or (>= '‪' and <= '‮') or (>= '⁦' and <= '⁩'))
            {
                return true;
            }
        }
        return false;
    }

    public static NormalizeResult Run(string input, NormalizeOptions options, WorkControl control)
    {
        control.Check();
        if (Uni.Len8(input) > NormalizerLimits.MaxInputBytes)
        {
            throw NormalizeException.Limited(LimitKind.Input);
        }
        if (options.Hints.Count > NormalizerLimits.MaxHints)
        {
            throw NormalizeException.Limited(LimitKind.Hints);
        }
        if (input.Trim().Length == 0 || Forbidden(input))
        {
            throw NormalizeException.Of(NormalizeErrorKind.InvalidInput);
        }
        var source = new SourceMap(input, control);
        var hints = source.Hints(options.Hints);
        var candidates = Classify.Collect(source, hints, control);
        var fallbackPolicy = options.AmbiguityPolicy == AmbiguityPolicy.Fallback;
        if (fallbackPolicy)
        {
            Symbols.Supplement(source, candidates, control);
        }
        var budget = new ResultBudget();
        budget.Charge(ResultSize);
        var segments = new List<Segment>();
        var issues = new List<Issue>();
        var fallbacks = new List<FallbackDiagnostic>();
        var cursor = 0;

        SourceRange Bytes(int start, int end) => new(source.Utf8At(start), source.Utf8At(end));

        void Verbatim(int start, int end)
        {
            if (start != end)
            {
                budget.Segment(source.Utf8At(end) - source.Utf8At(start));
                segments.Add(new Segment(Bytes(start, end), SegmentKind.Verbatim, input[start..end], "source.verbatim"));
            }
        }

        foreach (var candidate in candidates)
        {
            control.Check();
            Rng range;
            try
            {
                range = source.Original(candidate.Range);
            }
            catch (NormalizeException)
            {
                throw NormalizeException.Of(NormalizeErrorKind.Internal);
            }
            if (range.Start < cursor || range.Start >= range.End || range.End > input.Length)
            {
                throw NormalizeException.Of(NormalizeErrorKind.Internal);
            }
            Verbatim(cursor, range.Start);
            SegmentKind kind;
            string ruleId;
            string text;
            if (candidate.Reading.Resolved is { } value)
            {
                (kind, ruleId, text) = Verbalize.Render(value);
            }
            else if (fallbackPolicy)
            {
                var request = candidate.Reading.Unresolved!;
                budget.Charge(DiagnosticSize);
                var leading = source.GraphemeBefore(range.Start) is { } before && Uni.Any(before, Uni.IsAlphanumeric);
                var trailing = false;
                if (range.End < input.Length)
                {
                    var length = Uni.GraphemeLength(input.AsSpan(range.End));
                    trailing = Uni.Any(input.Substring(range.End, length), Uni.IsAlphanumeric);
                }
                var padding = (leading ? 1 : 0) + (trailing ? 1 : 0);
                var maximum = budget.TextAllowance() - padding;
                if (maximum < 0)
                {
                    throw NormalizeException.Limited(LimitKind.Result);
                }
                var recognition = source.Text[candidate.Range.Start..candidate.Range.End];
                var (rendered, strategy) = request.Render(recognition, maximum, control);
                if (leading)
                {
                    rendered = " " + rendered;
                }
                if (trailing)
                {
                    rendered += " ";
                }
                fallbacks.Add(new FallbackDiagnostic(Bytes(range.Start, range.End), request.Class, request.Reason,
                    request.Category, strategy));
                (kind, ruleId, text) = (SegmentKind.Fallback, "source.fallback", rendered);
            }
            else
            {
                var request = candidate.Reading.Unresolved!;
                var category = request.Category ?? throw NormalizeException.Of(NormalizeErrorKind.Internal);
                budget.Issue();
                issues.Add(new Issue(Bytes(range.Start, range.End), category, Explanation(category)));
                (kind, ruleId, text) = (SegmentKind.Unresolved, "source.unresolved", input[range.Start..range.End]);
            }
            budget.Segment(Uni.Len8(text));
            segments.Add(new Segment(Bytes(range.Start, range.End), kind, text, ruleId));
            cursor = range.End;
        }
        Verbatim(cursor, input.Length);
        control.Check();
        if (options.AmbiguityPolicy == AmbiguityPolicy.Reject && issues.Count > 0)
        {
            throw new NormalizeException(NormalizeErrorKind.Unresolved, issues: issues);
        }
        var normalized = new System.Text.StringBuilder();
        foreach (var segment in segments)
        {
            control.Check();
            normalized.Append(segment.Text);
        }
        return new NormalizeResult(normalized.ToString(), issues.Count == 0, segments, issues, fallbacks);
    }
}
