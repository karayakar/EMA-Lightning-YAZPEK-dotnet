// src/classify/mod.rs portu
namespace NormalizerTr;

/// <summary>Rust Reading: Resolved(Value) ya da Unresolved(Request).</summary>
internal sealed class Reading
{
    Reading(Value? resolved, FallbackRequest? unresolved)
    {
        Resolved = resolved;
        Unresolved = unresolved;
    }

    public Value? Resolved { get; }
    public FallbackRequest? Unresolved { get; }

    public static Reading Of(Value value) => new(value, null);
    public static Reading Of(FallbackRequest request) => new(null, request);
}

internal sealed record Candidate(Rng Range, Reading Reading);

internal static class Classify
{
    static void Push(List<Candidate> candidates, Claim claim, Reading reading)
    {
        if (candidates.Count == NormalizerLimits.MaxCandidates)
        {
            throw NormalizeException.Limited(LimitKind.Candidates);
        }
        candidates.Add(new Candidate(claim.Range, reading));
    }

    public static List<Candidate> Collect(SourceMap source, List<InternalHint> hints, WorkControl control)
    {
        var text = source.Text;
        var tokens = Scan.Tokens(source, control);
        var phoneLike = Scan.Phones(text, tokens, control);
        var bounds = new Boundaries(tokens, phoneLike);
        var ctx = new ReadContext(text, tokens);
        var candidates = new List<Candidate>();
        var index = 0;
        var hintIndex = 0;
        var roleUntil = 0;
        while (index < tokens.Count)
        {
            control.Check();
            if (hintIndex < hints.Count && hints[hintIndex].Range.Start <= tokens[index].Range.Start)
            {
                var hint = hints[hintIndex];
                Claim? detected = Readers.Read(ctx, index, bounds, index < roleUntil) is { } found
                    ? bounds.Claim(index, found.End)
                    : null;
                var claim = bounds.HintClaim(hint, detected);
                var hinted = text[claim.Range.Start..claim.Range.End];
                var value = Readers.Hint(hinted, hint.Kind)
                            ?? throw NormalizeException.Of(NormalizeErrorKind.InvalidHint);
                if (value is RomanValue)
                {
                    roleUntil = ctx.RomanAnchorEnd(index) ?? roleUntil;
                }
                index = claim.Next;
                Push(candidates, claim, Reading.Of(value));
                hintIndex++;
                continue;
            }
            if (hintIndex < hints.Count && hints[hintIndex].Range.Start < tokens[index].Range.End)
            {
                throw NormalizeException.Of(NormalizeErrorKind.InvalidHint);
            }
            if (Readers.Read(ctx, index, bounds, index < roleUntil) is { } read)
            {
                var claim = bounds.Claim(index, read.End);
                if (Boundaries.OverlapsHint(hints, claim.Range))
                {
                    throw NormalizeException.Of(NormalizeErrorKind.InvalidHint);
                }
                if (read.Reading.Resolved is RomanValue)
                {
                    roleUntil = ctx.RomanAnchorEnd(index) ?? roleUntil;
                }
                index = claim.Next;
                Push(candidates, claim, read.Reading);
            }
            else
            {
                index++;
            }
        }
        if (hintIndex != hints.Count)
        {
            throw NormalizeException.Of(NormalizeErrorKind.InvalidHint);
        }
        return candidates;
    }
}
