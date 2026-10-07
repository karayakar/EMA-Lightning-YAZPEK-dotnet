// api.py portu (CPU, ONNX Runtime). Bir Ema oluştur, tüm program boyunca tut.
//   SayAsync    : bir metnin tamamı, hazır olunca
//   SayManyAsync: metin listesi, ortak batch'lerde, sırayla
//   StreamAsync : parça parça; ilk parça 1 sn, sonrakiler 4 sn
// Çok sayıda iş parçacığından aynı anda çağrılabilir; hepsi Playhead üzerinden aynı modeli paylaşır.
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text.Json;

namespace EmaLightning;

/// <summary>One spoken text: float32 audio in [-1, 1] at SampleRate, and the seed that made it.</summary>
public sealed record Speech(float[] Audio, int SampleRate, double Duration, long Seed);

/// <summary>Ses dönüştürme ayarı; Pitch = Formant = 1 ise ses olduğu gibi kalır.</summary>
internal readonly record struct Voice(double Pitch, double Formant, VoiceMethod Method);

public sealed class Ema : IDisposable
{
    const string Probe = "Bugün hava çok güzel, yarın da yağmur yağacakmış; toplantı öğleden sonra başlayacak.";

    readonly Engine engine;
    readonly Frontend frontend;
    readonly Playhead playhead;
    readonly Expressions expressions;
    volatile int batchSize;

    /// <param name="modelDirectory">text.onnx, sound.onnx, decoder.onnx ve ema_meta.json klasörü</param>
    public Ema(string modelDirectory)
    {
        engine = new Engine(modelDirectory);
        frontend = new Frontend(engine.Vocab);
        playhead = new Playhead(engine, () => batchSize > 0 ? batchSize : BestBatchSize());
        // isteğe bağlı ifade klipleri: <model>\expressions\<kategori>_NN.wav
        expressions = Expressions.Load(Path.Combine(modelDirectory, "expressions"), engine.SampleRate);
    }

    /// <summary>Kullanılabilir ifade kategorileri (etiket adları).</summary>
    public IReadOnlyCollection<string> ExpressionNames => expressions.Names;

    /// <summary>
    /// The smallest batch that reaches 90% of this CPU's best throughput (1/2/4/8), measured once and cached.
    /// Önbellek: $XDG_CACHE_HOME (yoksa ~/.cache)/ema_lightning/batch_size_onnx_v1.json — PyTorch ölçümüyle karışmasın diye ayrı dosya.
    /// </summary>
    public int BestBatchSize()
    {
        // Playhead döngüsü engine.Lock altında buraya gelir; kilidi aynı sırayla almak kilitlenmeyi önler (Monitor yeniden girişli)
        lock (engine.Lock)
        {
            if (batchSize > 0)
            {
                return batchSize;
            }
            var key = $"cpu-{Environment.ProcessorCount}";
            var path = CachePath();
            var cached = File.Exists(path)
                ? JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(path)) ?? new()
                : new Dictionary<string, int>();
            if (!cached.ContainsKey(key))
            {
                var rates = new Dictionary<int, double>();
                foreach (var size in new[] { 1, 2, 4, 8 })
                {
                    rates[size] = Throughput(size);
                }
                var best = rates.Values.Max();
                cached[key] = rates.Where(r => r.Value >= 0.9 * best).Min(r => r.Key);
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                File.WriteAllText(path, JsonSerializer.Serialize(cached));
            }
            batchSize = cached[key];
            return batchSize;
        }
    }

    /// <summary>Speech for one text. With path, also writes a 16-bit WAV.</summary>
    public async Task<Speech> SayAsync(string text, double speed = 1.0, long? seed = null, int sampleRate = Engine.Rate,
        double pitch = 1.0, double formant = 1.0, VoiceMethod method = VoiceMethod.Harvest, string? path = null,
        CancellationToken cancellationToken = default)
    {
        Check(speed, seed, sampleRate, pitch, formant);
        var actual = seed ?? RandomSeed();
        var request = Submit(text, speed, actual, Engine.Window);
        var speech = await Collect(request, sampleRate, new Voice(pitch, formant, method), actual, cancellationToken);
        if (path != null)
        {
            Wav.Write(path, speech.Audio, sampleRate);
        }
        return speech;
    }

    /// <summary>One Speech per text, in order, made in shared batches. With path, a folder of 0.wav, 1.wav, …</summary>
    public async Task<List<Speech>> SayManyAsync(IReadOnlyList<string> texts, double speed = 1.0, long? seed = null,
        int sampleRate = Engine.Rate, double pitch = 1.0, double formant = 1.0, VoiceMethod method = VoiceMethod.Harvest,
        string? path = null, CancellationToken cancellationToken = default)
    {
        Check(speed, seed, sampleRate, pitch, formant);
        var seeds = texts.Select(_ => seed ?? RandomSeed()).ToList();
        var requests = texts.Select((t, i) => Submit(t, speed, seeds[i], Engine.Window)).ToList();
        var output = new List<Speech>(texts.Count);
        try
        {
            for (var i = 0; i < requests.Count; i++)
            {
                output.Add(await Collect(requests[i], sampleRate, new Voice(pitch, formant, method), seeds[i],
                    cancellationToken));
            }
        }
        finally
        {
            foreach (var request in requests)
            {
                request.Cancel();
            }
        }
        if (path != null)
        {
            Directory.CreateDirectory(path);
            var width = Math.Max(output.Count - 1, 0).ToString().Length;
            for (var i = 0; i < output.Count; i++)
            {
                Wav.Write(Path.Combine(path, $"{i.ToString().PadLeft(width, '0')}.wav"), output[i].Audio, sampleRate);
            }
        }
        return output;
    }

    /// <summary>
    /// float32 chunks as they are made: one second first, then four seconds at a time.
    /// Döngüden erken çıkmak (ya da iptal) o metnin kalan işini düşürür. İş ilk parça istendiğinde kuyruğa girer.
    /// </summary>
    public IAsyncEnumerable<float[]> StreamAsync(string text, double speed = 1.0, long? seed = null,
        int sampleRate = Engine.Rate, double pitch = 1.0, double formant = 1.0, VoiceMethod method = VoiceMethod.Harvest,
        CancellationToken cancellationToken = default)
    {
        Check(speed, seed, sampleRate, pitch, formant);
        return StreamCore(text, speed, seed ?? RandomSeed(), sampleRate, new Voice(pitch, formant, method),
            cancellationToken);
    }

    async IAsyncEnumerable<float[]> StreamCore(string text, double speed, long seed, int sampleRate, Voice voice,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var request = Submit(text, speed, seed, Engine.FirstWindow, stream: true);
        await foreach (var chunk in Receive(request, engine.SampleRate, sampleRate, voice, cancellationToken))
        {
            yield return chunk;
        }
    }

    Request Submit(string text, double speed, long seed, int first, bool stream = false) =>
        playhead.Submit(Pieces(text, speed, seed, stream), speed, first);

    List<Piece> Pieces(string text, double speed, long seed, bool stream = false)
    {
        // Etiket/emoji noktalarından bölünür: metin kısımları modele, ifadeler klip olarak (Clip) aynı sırayla
        var pieces = new List<(string Piece, double Pause, float[]? Clip)>();
        foreach (var (segment, clip) in expressions.Split(text, seed))
        {
            if (clip != null)
            {
                pieces.Add(("", 0.0, clip));
            }
            else
            {
                pieces.AddRange(Chunker.Chunk(frontend.Process(segment!), speed).Select(p => (p.Piece, p.Pause, (float[]?)null)));
            }
        }
        if (stream && pieces.Count > 0 && pieces[0].Clip == null)
        {
            // Stream: ilk kelime ayrı parça (ilk ses hızlı gelsin), ilk cümlenin kalanı ayrı; ikisi de tek başına
            // düşünülür (Solo), aralarında duraklama yok. Sonraki cümleler her zamanki gibi batch'lenir.
            var (first, pause, _) = pieces[0];
            var space = first.IndexOf(' ');
            if (space > 0)
            {
                pieces.RemoveAt(0);
                pieces.Insert(0, (first[(space + 1)..], pause, null));
                pieces.Insert(0, (Chunker.Finish(first[..space]), 0.0, null));
            }
        }
        var solo = stream ? Math.Min(2, pieces.Count) : 0;
        var result = new List<Piece>(pieces.Count);
        for (var i = 0; i < pieces.Count; i++)
        {
            // Python: (seed * 1_000_003 + i) % 2**63
            var mixed = (ulong)(((UInt128)(ulong)seed * 1_000_003 + (UInt128)i) % ((UInt128)1 << 63));
            var piece = pieces[i].Clip != null
                ? new Piece { Text = "", Pause = 0, Seed = mixed, Ids = Array.Empty<long>(), Cw = Array.Empty<long>(),
                    Wstart = Array.Empty<long>(), Fw = Array.Empty<long>(), Clip = pieces[i].Clip }
                : engine.Piece(pieces[i].Piece, pieces[i].Pause, mixed);
            piece.Solo = i < solo;
            result.Add(piece);
        }
        return result;
    }

    async Task<Speech> Collect(Request request, int sampleRate, Voice voice, long seed,
        CancellationToken cancellationToken)
    {
        var chunks = new List<float[]>();
        await foreach (var chunk in Receive(request, engine.SampleRate, sampleRate, voice, cancellationToken))
        {
            chunks.Add(chunk);
        }
        var audio = chunks.SelectMany(c => c).ToArray();
        return new Speech(audio, sampleRate, (double)audio.Length / sampleRate, seed);
    }

    /// <summary>
    /// Chunks from a request's outbox, voice-shifted at 48 kHz, then resampled. If the caller stops waiting, its work is dropped.
    /// </summary>
    static async IAsyncEnumerable<float[]> Receive(Request request, int sourceRate, int sampleRate, Voice voice,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        // 24 kHz modeller (erkek) önce 48 kHz'e yükseltilir; sonrası (dönüştürücüler, Resampler) 48 kHz zinciri
        var upsampler = sourceRate == Engine.Rate ? null : new Upsampler2();
        var identity = voice.Pitch == 1.0 && voice.Formant == 1.0;
        // B yolu pencere pencere akar; WORLD (A yolu) cümlenin tamamını (sesi + duraklaması) PieceEnd'de işler
        var shifter = voice.Method == VoiceMethod.Psola ? VoiceShifter.Create(voice.Pitch, voice.Formant) : null;
        var world = !identity && voice.Method != VoiceMethod.Psola;
        var sentence = new List<float[]>();
        var resampler = new Resampler(sampleRate);
        try
        {
            while (true)
            {
                var item = await request.Outbox.Reader.ReadAsync(cancellationToken);
                if (ReferenceEquals(item, Request.Done))
                {
                    break;
                }
                if (item is Exception error)
                {
                    ExceptionDispatchInfo.Capture(error).Throw();
                }
                if (ReferenceEquals(item, Request.PieceEnd))
                {
                    if (world && sentence.Count > 0)
                    {
                        var shiftedSentence = resampler.Push(WorldShifter.Process(sentence.SelectMany(c => c).ToArray(),
                            voice.Pitch, voice.Formant, voice.Method));
                        sentence.Clear();
                        if (shiftedSentence.Length > 0)
                        {
                            yield return shiftedSentence;
                        }
                    }
                    continue;
                }
                var audio = (float[])item;
                if (upsampler != null)
                {
                    audio = upsampler.Push(audio);
                }
                if (world)
                {
                    sentence.Add(audio);
                    continue;
                }
                var output = resampler.Push(shifter != null ? shifter.Push(audio) : audio);
                if (output.Length > 0)
                {
                    yield return output;
                }
            }
            if (upsampler != null)
            {
                var rest = upsampler.Flush();
                if (world)
                {
                    sentence.Add(rest);
                }
                else
                {
                    var restOut = resampler.Push(shifter != null ? shifter.Push(rest) : rest);
                    if (restOut.Length > 0)
                    {
                        yield return restOut;
                    }
                }
            }
            if (world && sentence.Count > 0)
            {
                var lastSentence = resampler.Push(WorldShifter.Process(sentence.SelectMany(c => c).ToArray(),
                    voice.Pitch, voice.Formant, voice.Method));
                if (lastSentence.Length > 0)
                {
                    yield return lastSentence;
                }
            }
            if (shifter != null)
            {
                var shifted = resampler.Push(shifter.Flush());
                if (shifted.Length > 0)
                {
                    yield return shifted;
                }
            }
            var tail = resampler.Flush();
            if (tail is { Length: > 0 })
            {
                yield return tail;
            }
        }
        finally
        {
            request.Cancel();
        }
    }

    double Throughput(int size)
    {
        var probe = Pieces(Probe, 1.0, 0);
        if (probe.Count != 1)
        {
            throw new InvalidOperationException("probe sentence must be a single piece");
        }
        var pieces = Enumerable.Range(0, size).Select(i => engine.Piece(probe[0].Text, 0.0, (ulong)i)).ToList();
        var best = double.MaxValue;
        lock (engine.Lock)
        {
            engine.Plan(pieces, 1.0);
            for (var pass = 0; pass < 3; pass++)
            {
                var watch = Stopwatch.StartNew();
                engine.Think(pieces);
                foreach (var span in Engine.Windows(pieces[0].Frames))
                {
                    engine.Decode(pieces.Select(p => (p, span)).ToList());
                }
                best = Math.Min(best, watch.Elapsed.TotalSeconds);
            }
        }
        return size * pieces[0].Frames / 25.0 / best;
    }

    static string CachePath()
    {
        var root = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
        if (string.IsNullOrEmpty(root))
        {
            root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
        }
        return Path.Combine(root, "ema_lightning", "batch_size_onnx_v1.json");
    }

    static long RandomSeed() => RandomNumberGenerator.GetInt32(int.MaxValue);

    static void Check(double speed, long? seed, int sampleRate, double pitch, double formant)
    {
        if (double.IsNaN(pitch) || pitch < VoiceShifter.MinPitch || pitch > VoiceShifter.MaxPitch)
        {
            throw new ArgumentException($"pitch must be a number from {VoiceShifter.MinPitch} to {VoiceShifter.MaxPitch}",
                nameof(pitch));
        }
        if (double.IsNaN(formant) || formant < VoiceShifter.MinFormant || formant > VoiceShifter.MaxFormant)
        {
            throw new ArgumentException(
                $"formant must be a number from {VoiceShifter.MinFormant} to {VoiceShifter.MaxFormant}", nameof(formant));
        }
        if (double.IsNaN(speed) || speed < 0.25 || speed > 4)
        {
            throw new ArgumentException("speed must be a number from 0.25 to 4", nameof(speed));
        }
        if (seed is < 0)
        {
            throw new ArgumentException("seed must be a non-negative integer", nameof(seed));
        }
        if (!Resampler.Rates.Contains(sampleRate))
        {
            throw new ArgumentException($"sample_rate must be one of {string.Join(", ", Resampler.Rates)}",
                nameof(sampleRate));
        }
    }

    public void Dispose() => engine.Dispose();
}
