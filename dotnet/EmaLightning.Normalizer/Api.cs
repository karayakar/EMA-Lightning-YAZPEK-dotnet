// normalizer-tr 0.4.0 (Rust) public API'sinin C# portu: src/api.rs
// Dışarıya verilen tüm aralıklar Rust'taki gibi özgün metnin UTF-8 bayt koordinatlarıdır.
using System.Diagnostics;

namespace NormalizerTr;

/// <summary>Original-source half-open UTF-8 byte range.</summary>
public readonly record struct SourceRange(int Start, int End);

/// <summary>How unresolved linguistic expressions are handled.</summary>
public enum AmbiguityPolicy
{
    Preserve,
    Reject,
    Fallback,
}

/// <summary>Explicit interpretation of a whole original-source span.</summary>
public enum HintKind
{
    Cardinal,
    Digits,
    Date,
    Time,
    Ordinal,
    Roman,
    Range,
    Telephone,
    Electronic,
}

/// <summary>Caller intent at original grapheme-safe byte coordinates.</summary>
public readonly record struct Hint(SourceRange Range, HintKind Kind);

/// <summary>Per-call options. Source-faithful fallback is opt-in.</summary>
public sealed class NormalizeOptions
{
    public AmbiguityPolicy AmbiguityPolicy { get; set; } = AmbiguityPolicy.Preserve;
    public List<Hint> Hints { get; set; } = new();
}

/// <summary>Semantic kind of a result segment.</summary>
public enum SegmentKind
{
    Verbatim,
    Unresolved,
    Cardinal,
    Ordinal,
    Decimal,
    Digits,
    Percent,
    Money,
    Unit,
    Date,
    Time,
    Abbreviation,
    Range,
    Telephone,
    Iban,
    Roman,
    Electronic,
    Symbol,
    Fallback,
}

/// <summary>Source family attempted before fallback rendering.</summary>
public enum FallbackClass
{
    Number,
    Date,
    Time,
    Percent,
    Quantity,
    Abbreviation,
    Identifier,
    Roman,
    Electronic,
    Expression,
    Symbol,
}

/// <summary>Why a primary reading was unavailable.</summary>
public enum FallbackReason
{
    MissingIntent,
    LeadingZeroes,
    InvalidForm,
    ProtectedIdentifier,
    UnsupportedForm,
    UnapprovedAbbreviation,
    UnhandledSymbol,
}

/// <summary>Applied source-faithful rendering strategy.</summary>
public enum FallbackStrategy
{
    PreferredNumber,
    PreferredDate,
    PreferredTime,
    SurfaceDate,
    SurfaceTime,
    Literal,
    UnicodeCodePoint,
}

/// <summary>Machine-readable reason for preserved linguistic work.</summary>
public enum IssueCategory
{
    Ambiguous,
    InvalidExpression,
    ProtectedIdentifier,
    Unsupported,
    UnknownAbbreviation,
}

/// <summary>Immutable provenance for one handled original-source fallback span.</summary>
public sealed record FallbackDiagnostic(
    SourceRange Range,
    FallbackClass AttemptedClass,
    FallbackReason Reason,
    IssueCategory? OriginalCategory,
    FallbackStrategy Strategy);

/// <summary>Non-sensitive diagnostic for an unresolved original-source range.</summary>
public sealed record Issue(SourceRange Range, IssueCategory Category, string Explanation);

/// <summary>One member of an ordered, contiguous original-source partition.</summary>
public sealed record Segment(SourceRange Range, SegmentKind Kind, string Text, string RuleId);

/// <summary>Owned immutable result. Completeness concerns TN work, not voice quality.</summary>
public sealed class NormalizeResult
{
    internal NormalizeResult(string normalizedText, bool complete, IReadOnlyList<Segment> segments,
        IReadOnlyList<Issue> issues, IReadOnlyList<FallbackDiagnostic> fallbacks)
    {
        NormalizedText = normalizedText;
        Complete = complete;
        Segments = segments;
        Issues = issues;
        Fallbacks = fallbacks;
    }

    public string NormalizedText { get; }
    public string Locale => "tr-TR";
    public string NormalizerId => Normalizer.Id;
    public bool Complete { get; }
    public IReadOnlyList<Segment> Segments { get; }
    public IReadOnlyList<Issue> Issues { get; }
    public IReadOnlyList<FallbackDiagnostic> Fallbacks { get; }
    public bool FallbackUsed => Fallbacks.Count > 0;
}

/// <summary>Resource limit which was exceeded.</summary>
public enum LimitKind
{
    Input,
    Hints,
    Candidates,
    Result,
}

/// <summary>Rust NormalizeError varyantları.</summary>
public enum NormalizeErrorKind
{
    InvalidInput,
    InvalidHint,
    InvalidConfiguration,
    LimitExceeded,
    Cancelled,
    Unresolved,
    Internal,
}

/// <summary>Explicit input, policy, resource, control, or engine failure (Rust NormalizeError).</summary>
public sealed class NormalizeException : Exception
{
    public NormalizeException(NormalizeErrorKind kind, LimitKind? limit = null, IReadOnlyList<Issue>? issues = null)
        : base(MessageOf(kind))
    {
        Kind = kind;
        Limit = limit;
        Issues = issues ?? Array.Empty<Issue>();
    }

    public NormalizeErrorKind Kind { get; }
    public LimitKind? Limit { get; }
    public IReadOnlyList<Issue> Issues { get; }

    // Rust Debug biçimi: "InvalidHint", "LimitExceeded(Input)" ...
    public string DebugName => Kind == NormalizeErrorKind.LimitExceeded ? $"LimitExceeded({Limit})" : Kind.ToString();

    static string MessageOf(NormalizeErrorKind kind) => kind switch
    {
        NormalizeErrorKind.InvalidInput => "input is empty, whitespace-only, or contains forbidden controls",
        NormalizeErrorKind.InvalidHint => "hint is not a valid whole-expression original-source range",
        NormalizeErrorKind.InvalidConfiguration => "built-in normalizer configuration is invalid",
        NormalizeErrorKind.LimitExceeded => "normalization resource limit exceeded",
        NormalizeErrorKind.Cancelled => "normalization was cancelled or its deadline expired",
        NormalizeErrorKind.Unresolved => "strict normalization contains unresolved linguistic work",
        _ => "normalization invariant failed",
    };

    internal static NormalizeException Of(NormalizeErrorKind kind) => new(kind);
    internal static NormalizeException Limited(LimitKind limit) => new(NormalizeErrorKind.LimitExceeded, limit);
}

/// <summary>Runtime-neutral cooperative control. Aynı nesneyi paylaşanlar iptal sinyalini paylaşır.</summary>
public sealed class WorkControl
{
    int cancelled;
    readonly long? deadline; // Stopwatch zaman damgası (monoton)

    /// <summary>Optional monotonic deadline as a timeout from now.</summary>
    public WorkControl(TimeSpan? timeout = null)
    {
        if (timeout is { } t)
        {
            deadline = Stopwatch.GetTimestamp() + (long)(t.TotalSeconds * Stopwatch.Frequency);
        }
    }

    public void Cancel() => Interlocked.Exchange(ref cancelled, 1);

    internal void Check()
    {
        if (Volatile.Read(ref cancelled) != 0 || (deadline is { } d && Stopwatch.GetTimestamp() >= d))
        {
            throw new NormalizeException(NormalizeErrorKind.Cancelled);
        }
    }
}

/// <summary>src/lib.rs sabitleri.</summary>
public static class NormalizerLimits
{
    public const int MaxInputBytes = 32 * 1024;
    public const int MaxHints = 256;
    public const int MaxCandidates = 4096;
    public const int MaxResultBytes = 512 * 1024;
}
