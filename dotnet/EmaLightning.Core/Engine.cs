// engine.py portu: konuşma parçaları → 48 kHz ses. Üç aşama: plan (metin + zaman çizelgesi),
// think (hizalayıcı + 4 adım, sound.onnx), decode (bir pencere latent → ses, decoder.onnx).
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;

namespace EmaLightning;

internal sealed class ModelMeta
{
    [JsonPropertyName("vocab")] public List<string> Vocab { get; set; } = new();
    [JsonPropertyName("times")] public List<double> Times { get; set; } = new();
    [JsonPropertyName("latent_dim")] public int LatentDim { get; set; }
    [JsonPropertyName("d")] public int D { get; set; }
    [JsonPropertyName("hop")] public int Hop { get; set; }
    [JsonPropertyName("sample_rate")] public int SampleRate { get; set; }
    [JsonPropertyName("latent_rate")] public int LatentRate { get; set; }

    public static ModelMeta Load(string path) =>
        JsonSerializer.Deserialize<ModelMeta>(File.ReadAllText(path))
        ?? throw new InvalidDataException($"invalid model metadata: {path}");
}

internal sealed class Piece
{
    public required string Text { get; init; }
    public required double Pause { get; init; }
    public required ulong Seed { get; init; }
    public required long[] Ids { get; init; }
    public required long[] Cw { get; init; } // her harfin kelimesi
    public required long[] Wstart { get; init; } // o kelimenin ilk harfi
    public float[]? H { get; set; }
    public float[]? Dur { get; set; }
    public long[]? Fw { get; set; } // her karenin kelimesi
    public float[]? Fp { get; set; } // karenin kelime içindeki konumu
    public float[]? Latents { get; set; }
    public List<(int Start, int End)>? Spans { get; set; } // planlanınca sabitlenen decoder pencereleri

    /// <summary>Stream başı: bu parça başka parçalarla aynı think batch'ine konmaz (ilk ses beklemesin).</summary>
    public bool Solo { get; set; }

    /// <summary>İfade klibi (modelin hızında hazır ses): plan/think/decode yapılmaz, sırası gelince olduğu gibi gönderilir.</summary>
    public float[]? Clip { get; init; }

    public int Letters => Ids.Length;
    public int Frames => Fw!.Length;
}

internal sealed class Engine : IDisposable
{
    /// <summary>Çıkış zincirinin hızı (ses dönüştürücüler + Resampler). 24 kHz modeller önce buna yükseltilir.</summary>
    public const int Rate = 48000;
    public const int FirstWindow = 25; // bir stream'in ilk penceresi 1 sn, ilk ses erken gelsin
    public const int Window = 100; // sonraki her pencere 4 sn
    public const int Context = 8; // pencerenin iki yanında decode edilen kare
    const int MaxWordFrames = 250;
    const int MaxFrames = 3000;

    readonly InferenceSession text;
    readonly InferenceSession sound;
    readonly InferenceSession decoder;
    readonly Dictionary<string, long> stoi;
    readonly int d;
    readonly int latentDim;
    readonly int steps;

    public Engine(string modelDirectory)
    {
        var meta = ModelMeta.Load(Path.Combine(modelDirectory, "ema_meta.json"));
        Vocab = meta.Vocab;
        stoi = new Dictionary<string, long>(StringComparer.Ordinal);
        for (var i = 0; i < meta.Vocab.Count; i++)
        {
            stoi.TryAdd(meta.Vocab[i], i);
        }
        d = meta.D;
        latentDim = meta.LatentDim;
        steps = meta.Times.Count;
        Hop = meta.Hop;
        SampleRate = meta.SampleRate > 0 ? meta.SampleRate : Rate;
        if (SampleRate != Rate && SampleRate * 2 != Rate)
        {
            throw new InvalidDataException($"unsupported model sample rate {SampleRate} (48000 or 24000)");
        }
        text = new InferenceSession(Path.Combine(modelDirectory, "text.onnx"));
        sound = new InferenceSession(Path.Combine(modelDirectory, "sound.onnx"));
        decoder = new InferenceSession(Path.Combine(modelDirectory, "decoder.onnx"));
    }

    public object Lock { get; } = new();
    public IReadOnlyList<string> Vocab { get; }
    public int Hop { get; }

    /// <summary>Modelin kendi örnekleme hızı (ema_meta.json): kadın 48000, erkek 24000.</summary>
    public int SampleRate { get; }

    public Piece Piece(string piece, double pause, ulong seed)
    {
        var ids = new long[piece.Length];
        for (var i = 0; i < piece.Length; i++)
        {
            ids[i] = stoi.TryGetValue(piece[i].ToString(), out var id) ? id : 1;
        }
        var starts = new List<int>();
        for (var i = 0; i < piece.Length; i++)
        {
            if (piece[i] != ' ' && (i == 0 || piece[i - 1] == ' '))
            {
                starts.Add(i);
            }
        }
        if (starts.Count == 0)
        {
            starts.Add(0);
        }
        var bounds = new List<int> { 0 };
        bounds.AddRange(starts.Skip(1));
        bounds.Add(piece.Length);
        var cw = new long[piece.Length];
        var wstart = new long[piece.Length];
        for (var w = 0; w + 1 < bounds.Count; w++)
        {
            for (var i = bounds[w]; i < bounds[w + 1]; i++)
            {
                cw[i] = w;
                wstart[i] = bounds[w];
            }
        }
        return new Piece { Text = piece, Pause = pause, Seed = seed, Ids = ids, Cw = cw, Wstart = wstart };
    }

    /// <summary>Durations for a batch, then each piece's exact frame timeline.</summary>
    public void Plan(IReadOnlyList<Piece> pieces, double speed)
    {
        var batch = pieces.Count;
        var length = pieces.Max(p => p.Letters);
        var ids = Stack(pieces.Select(p => p.Ids).ToList(), length, 1, 0L);
        var mask = ids.Select(id => id != 0).ToArray();
        var outputs = Run(text, new[] { "ids", "mask" }, new[] { Tensor(ids, batch, length), Tensor(mask, batch, length) },
            new[] { "h", "dur" });
        var h = outputs[0];
        var dur = outputs[1];
        var divisor = (float)speed;
        for (var i = 0; i < dur.Length; i++)
        {
            dur[i] /= divisor;
        }
        var word = Stack(pieces.Select(p => p.Cw).ToList(), length, 1, 0L);
        for (var b = 0; b < batch; b++)
        {
            // torch: zeros_like(dur).scatter_add_(1, word, dur).round().clamp(1, MAX_WORD_FRAMES)
            var sums = new float[length];
            for (var l = 0; l < length; l++)
            {
                sums[word[b * length + l]] += dur[b * length + l];
            }
            var piece = pieces[b];
            var words = (int)piece.Cw[^1] + 1;
            var counts = new int[words];
            for (var w = 0; w < words; w++)
            {
                counts[w] = (int)Math.Clamp(MathF.Round(sums[w], MidpointRounding.ToEven), 1, MaxWordFrames);
            }
            var frames = Math.Min(counts.Sum(), MaxFrames);
            var fw = new long[frames];
            var fp = new float[frames];
            var f = 0;
            var before = 0;
            for (var w = 0; w < words && f < frames; w++)
            {
                for (var k = 0; k < counts[w] && f < frames; k++, f++)
                {
                    fw[f] = w;
                    fp[f] = (float)((double)(f - before) / counts[w]);
                }
                before += counts[w];
            }
            piece.H = h.AsSpan(b * length * d, piece.Letters * d).ToArray();
            piece.Dur = dur.AsSpan(b * length, piece.Letters).ToArray();
            piece.Fw = fw;
            piece.Fp = fp;
        }
    }

    /// <summary>Latents for a batch, each piece from its own seeded noise.</summary>
    public void Think(IReadOnlyList<Piece> pieces)
    {
        var batch = pieces.Count;
        var length = pieces.Max(p => p.Letters);
        var frames = pieces.Max(p => p.Frames);
        var h = Stack(pieces.Select(p => p.H!).ToList(), length, d, 0f);
        var dur = Stack(pieces.Select(p => p.Dur!).ToList(), length, 1, 0f);
        var mask = Stack(pieces.Select(p => Enumerable.Repeat(true, p.Letters).ToArray()).ToList(), length, 1, false);
        var cw = Stack(pieces.Select(p => p.Cw).ToList(), length, 1, -1L);
        var wstart = Stack(pieces.Select(p => p.Wstart).ToList(), length, 1, 0L);
        var fw = Stack(pieces.Select(p => p.Fw!).ToList(), frames, 1, -1L);
        var fp = Stack(pieces.Select(p => p.Fp!).ToList(), frames, 1, 0f);
        var fmask = Stack(pieces.Select(p => Enumerable.Repeat(true, p.Frames).ToArray()).ToList(), frames, 1, false);
        // noise [B, steps, T, latent]: her adım için parçanın kendi gürültüsü, T'ye sıfırla doldurulur
        var noise = new float[batch * steps * frames * latentDim];
        for (var b = 0; b < batch; b++)
        {
            var piece = pieces[b];
            var own = Noise(piece);
            for (var s = 0; s < steps; s++)
            {
                Array.Copy(own, s * piece.Frames * latentDim, noise, ((b * steps + s) * frames) * latentDim,
                    piece.Frames * latentDim);
            }
        }
        var latents = Run(sound,
            new[] { "h", "dur", "mask", "cw", "wstart", "fw", "fp", "fmask", "noise" },
            new[]
            {
                Tensor(h, batch, length, d), Tensor(dur, batch, length), Tensor(mask, batch, length),
                Tensor(cw, batch, length), Tensor(wstart, batch, length), Tensor(fw, batch, frames),
                Tensor(fp, batch, frames), Tensor(fmask, batch, frames), Tensor(noise, batch, steps, frames, latentDim),
            },
            new[] { "latents" })[0];
        for (var b = 0; b < batch; b++)
        {
            pieces[b].Latents = latents.AsSpan(b * frames * latentDim, pieces[b].Frames * latentDim).ToArray();
        }
    }

    /// <summary>Audio for a batch of (piece, window), each cut to its own window.</summary>
    public List<float[]> Decode(IReadOnlyList<(Piece Piece, (int Start, int End) Span)> items)
    {
        var spans = items.Select(x => (Start: Math.Max(0, x.Span.Start - Context),
            End: Math.Min(x.Piece.Frames, x.Span.End + Context))).ToList();
        var padded = spans.Max(s => s.End - s.Start); // CPU'da sabit decode boyu yok (engine.py decode_sizes)
        var batch = items.Count;
        var z = new float[batch * padded * latentDim];
        var lengths = new long[batch];
        for (var i = 0; i < batch; i++)
        {
            var (a, b) = spans[i];
            Array.Copy(items[i].Piece.Latents!, a * latentDim, z, i * padded * latentDim, (b - a) * latentDim);
            lengths[i] = b - a;
        }
        var audio = Run(decoder, new[] { "z", "lengths" }, new[] { Tensor(z, batch, padded, latentDim), Tensor(lengths, batch) },
            new[] { "audio" })[0];
        var samples = padded * Hop;
        var result = new List<float[]>(batch);
        for (var i = 0; i < batch; i++)
        {
            var (s, e) = items[i].Span;
            var a = spans[i].Start;
            result.Add(audio.AsSpan(i * samples + (s - a) * Hop, (e - s) * Hop).ToArray());
        }
        return result;
    }

    /// <summary>(start, end) of every decoded window: `first` frames first, then four seconds each.</summary>
    public static List<(int Start, int End)> Windows(int frames, int first = Window)
    {
        var spans = new List<(int, int)>();
        var s = 0;
        while (s < frames)
        {
            var e = Math.Min(frames, s + (s == 0 ? first : Window));
            spans.Add((s, e));
            s = e;
        }
        return spans;
    }

    float[] Noise(Piece piece)
    {
        // .NET kendi RNG'si (kullanıcı kararı): aynı seed → aynı ses, PyTorch ile aynı değil
        var random = new NormalRandom(piece.Seed);
        var noise = new float[steps * piece.Frames * latentDim];
        for (var i = 0; i < noise.Length; i++)
        {
            noise[i] = random.Next();
        }
        return noise;
    }

    static T[] Stack<T>(IReadOnlyList<T[]> rows, int size, int width, T fill)
    {
        var output = new T[rows.Count * size * width];
        Array.Fill(output, fill);
        for (var i = 0; i < rows.Count; i++)
        {
            Array.Copy(rows[i], 0, output, i * size * width, rows[i].Length);
        }
        return output;
    }

    static OrtValue Tensor<T>(T[] data, params long[] shape) where T : unmanaged =>
        OrtValue.CreateTensorValueFromMemory(data, shape);

    static List<float[]> Run(InferenceSession session, string[] names, OrtValue[] inputs, string[] outputNames)
    {
        try
        {
            using var runOptions = new RunOptions();
            using var outputs = session.Run(runOptions, names, inputs, outputNames);
            return outputs.Select(o => o.GetTensorDataAsSpan<float>().ToArray()).ToList();
        }
        finally
        {
            foreach (var input in inputs)
            {
                input.Dispose();
            }
        }
    }

    public void Dispose()
    {
        text.Dispose();
        sound.Dispose();
        decoder.Dispose();
    }
}

/// <summary>Deterministik standart normal üreteç: SplitMix64 tohumlu xoshiro256** + Box-Muller.</summary>
internal sealed class NormalRandom
{
    ulong s0, s1, s2, s3;
    float? spare;

    public NormalRandom(ulong seed)
    {
        s0 = SplitMix(ref seed);
        s1 = SplitMix(ref seed);
        s2 = SplitMix(ref seed);
        s3 = SplitMix(ref seed);
    }

    static ulong SplitMix(ref ulong x)
    {
        var z = x += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    ulong NextULong()
    {
        var result = BitOperations.RotateLeft(s1 * 5, 7) * 9;
        var t = s1 << 17;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = BitOperations.RotateLeft(s3, 45);
        return result;
    }

    double NextDouble() => (NextULong() >> 11) * (1.0 / (1UL << 53));

    public float Next()
    {
        if (spare is { } value)
        {
            spare = null;
            return value;
        }
        var u1 = 1.0 - NextDouble(); // (0, 1]
        var u2 = NextDouble();
        var radius = Math.Sqrt(-2.0 * Math.Log(u1));
        spare = (float)(radius * Math.Sin(2 * Math.PI * u2));
        return (float)(radius * Math.Cos(2 * Math.PI * u2));
    }
}
