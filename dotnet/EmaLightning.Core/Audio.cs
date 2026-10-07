// audio.py portu: 48 kHz'ten örnekleme hızı dönüşümü (pencere pencere, durum taşıyarak) ve WAV yazımı.
namespace EmaLightning;

/// <summary>48 kHz in, 48 kHz / factor out, through a linear-phase windowed-sinc low-pass.</summary>
internal sealed class Resampler
{
    public static readonly int[] Rates = { 48000, 24000, 16000, 8000 };

    readonly int factor;
    readonly int half;
    float[]? taps;
    float[] buffer = Array.Empty<float>();
    bool started;
    long start; // buffer[0]'ın mutlak giriş indeksi; sinyalin önünde sıfırlar var
    long next; // bir sonraki çıkış örneğinin merkezindeki mutlak giriş indeksi
    long end; // son gerçek örneğin bir ötesi

    public Resampler(int rate)
    {
        factor = Engine.Rate / rate;
        half = 32 * factor;
        start = -half;
    }

    public float[] Push(float[] chunk)
    {
        if (factor == 1)
        {
            return chunk;
        }
        if (!started)
        {
            started = true;
            buffer = new float[half];
            var length = 2 * half + 1;
            var cutoff = 0.45 / factor;
            var raw = new double[length];
            var sum = 0.0;
            for (var i = 0; i < length; i++)
            {
                double n = i - half;
                var x = 2 * cutoff * n;
                var sinc = x == 0 ? 1.0 : Math.Sin(Math.PI * x) / (Math.PI * x);
                // torch.blackman_window(N, periodic=False)
                var window = 0.42 - 0.5 * Math.Cos(2 * Math.PI * i / (length - 1))
                             + 0.08 * Math.Cos(4 * Math.PI * i / (length - 1));
                raw[i] = 2 * cutoff * sinc * window;
                sum += raw[i];
            }
            taps = raw.Select(t => (float)(t / sum)).ToArray();
        }
        buffer = Concat(buffer, chunk);
        end += chunk.Length;
        return Emit(end - 1 - half);
    }

    public float[]? Flush()
    {
        if (factor == 1 || !started)
        {
            return null;
        }
        buffer = Concat(buffer, new float[half + factor]);
        return Emit(end - 1);
    }

    /// <summary>Every output whose centre is at or before input index `last`.</summary>
    float[] Emit(long last)
    {
        if (last < next)
        {
            return Array.Empty<float>();
        }
        var count = (int)((last - next) / factor + 1);
        var a = (int)(next - half - start);
        var output = new float[count];
        for (var k = 0; k < count; k++)
        {
            var offset = a + k * factor;
            var acc = 0f;
            for (var j = 0; j < taps!.Length; j++)
            {
                acc += buffer[offset + j] * taps[j];
            }
            output[k] = acc;
        }
        next += (long)count * factor;
        var drop = (int)(next - half - start);
        buffer = buffer[drop..];
        start += drop;
        return output;
    }

    static float[] Concat(float[] a, float[] b)
    {
        var result = new float[a.Length + b.Length];
        a.CopyTo(result, 0);
        b.CopyTo(result, a.Length);
        return result;
    }
}

/// <summary>
/// 24 → 48 kHz akış halinde ×2 yükseltme (sıfır ekleme + Blackman pencereli sinc, kaynak Nyquist'in %95'i).
/// Durum pencereden pencereye taşınır; çıktı, giriş uzunluğunun tam iki katıdır (Flush ile).
/// </summary>
internal sealed class Upsampler2
{
    const int Half = 32; // her iki yanda giriş örneği
    readonly float[] taps = new float[4 * Half + 1]; // çıkış hızında, merkez 2·Half
    float[] buffer = Array.Empty<float>();
    long start; // buffer[0]'ın mutlak giriş indeksi
    long end; // girişin sonu (gerçek)
    long next; // sıradaki çıkış indeksi

    public Upsampler2()
    {
        var length = taps.Length;
        const double cutoff = 0.95 * 0.25; // devir/örnek (48 kHz'te) → ~11.4 kHz
        var sum = 0.0;
        var raw = new double[length];
        for (var i = 0; i < length; i++)
        {
            double n = i - 2 * Half;
            var x = 2 * cutoff * n;
            var sinc = x == 0 ? 1.0 : Math.Sin(Math.PI * x) / (Math.PI * x);
            var window = 0.42 - 0.5 * Math.Cos(2 * Math.PI * i / (length - 1)) + 0.08 * Math.Cos(4 * Math.PI * i / (length - 1));
            raw[i] = 2 * cutoff * sinc * window;
            sum += raw[i];
        }
        for (var i = 0; i < length; i++)
        {
            taps[i] = (float)(raw[i] / sum * 2); // sıfır eklemenin yarıya düşürdüğü kazanç geri verilir
        }
    }

    public float[] Push(float[] chunk)
    {
        Append(chunk);
        end += chunk.Length;
        // çıkış i için giriş k ≤ (i + 2·Half) / 2 gerekir
        return Emit(2 * end - 2 - 2 * Half);
    }

    public float[] Flush()
    {
        Append(new float[Half + 1]);
        return Emit(2 * end - 1);
    }

    void Append(float[] chunk)
    {
        var merged = new float[buffer.Length + chunk.Length];
        buffer.CopyTo(merged, 0);
        chunk.CopyTo(merged, buffer.Length);
        buffer = merged;
    }

    float Input(long k) => k < start || k - start >= buffer.Length ? 0f : buffer[k - start];

    float[] Emit(long last)
    {
        if (last < next)
        {
            return Array.Empty<float>();
        }
        var output = new float[last - next + 1];
        for (var o = 0; o < output.Length; o++)
        {
            var i = next + o;
            var acc = 0f;
            for (var k = (long)Math.Ceiling((i - 2.0 * Half) / 2); k <= (i + 2 * Half) / 2; k++)
            {
                acc += Input(k) * taps[i - 2 * k + 2 * Half];
            }
            output[o] = acc;
        }
        next = last + 1;
        var keep = (long)Math.Ceiling((next - 2.0 * Half) / 2) - 1;
        if (keep > start)
        {
            var drop = (int)Math.Min(keep - start, buffer.Length);
            buffer = buffer[drop..];
            start += drop;
        }
        return output;
    }
}

public static class Wav
{
    /// <summary>16-bit PCM mono WAV (audio.py write_wav).</summary>
    public static void Write(string path, float[] audio, int rate)
    {
        using var stream = File.Create(path);
        Write(stream, audio, rate);
    }

    public static void Write(Stream stream, float[] audio, int rate)
    {
        using var writer = new BinaryWriter(stream, System.Text.Encoding.ASCII, leaveOpen: true);
        var dataBytes = audio.Length * 2;
        writer.Write("RIFF"u8);
        writer.Write(36 + dataBytes);
        writer.Write("WAVE"u8);
        writer.Write("fmt "u8);
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(rate);
        writer.Write(rate * 2);
        writer.Write((short)2);
        writer.Write((short)16);
        writer.Write("data"u8);
        writer.Write(dataBytes);
        foreach (var sample in audio)
        {
            // numpy: (clip * 32767).round() — yarım değerler çifte yuvarlanır
            writer.Write((short)Math.Round(Math.Clamp(sample, -1f, 1f) * 32767.0, MidpointRounding.ToEven));
        }
    }
}
