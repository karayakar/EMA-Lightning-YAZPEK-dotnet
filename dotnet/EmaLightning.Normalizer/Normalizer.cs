// src/normalizer.rs portu
namespace NormalizerTr;

/// <summary>Immutable, thread-safe normalizer.</summary>
public sealed class Normalizer
{
    /// <summary>Diagnostic identity: kaynak alınan Rust sürümü.</summary>
    public const string Id = "normalizer-tr/0.4.0";

    public string NormalizerId => Id;

    /// <summary>Normalize synchronously with default, uncancelled work control.</summary>
    public NormalizeResult Normalize(string input, NormalizeOptions? options = null) =>
        NormalizeControlled(input, options ?? new NormalizeOptions(), new WorkControl());

    /// <summary>Normalize with cooperative cancellation and an optional deadline.</summary>
    public NormalizeResult NormalizeControlled(string input, NormalizeOptions options, WorkControl control) =>
        Pipeline.Run(input, options, control);
}
