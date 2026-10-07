// Konuşmacı tonu dönüştürme (B yolu, bağımlılıksız): 48 kHz'te, akış halinde (Push/Flush).
//   F0      : YIN ile perde takibi + TD-PSOLA (formantları korur)
//   Formant : STFT çerçevelerinde LPC zarfının frekans ekseninde bükülmesi
// pitch = 1 ve formant = 1 ise hiç oluşturulmaz; ses olduğu gibi geçer.
namespace EmaLightning;

internal sealed class VoiceShifter
{
    public const double MinPitch = 0.5, MaxPitch = 2.0, MinFormant = 0.7, MaxFormant = 1.4;

    readonly Psola? psola;
    readonly FormantWarp? warp;

    VoiceShifter(double pitch, double formant)
    {
        psola = pitch != 1.0 ? new Psola(pitch) : null;
        warp = formant != 1.0 ? new FormantWarp(formant) : null;
    }

    /// <summary>Kimlik dönüşümünde null döner.</summary>
    public static VoiceShifter? Create(double pitch, double formant) =>
        pitch == 1.0 && formant == 1.0 ? null : new VoiceShifter(pitch, formant);

    public float[] Push(float[] chunk)
    {
        var y = psola != null ? psola.Push(chunk) : chunk;
        return warp != null ? warp.Push(y) : y;
    }

    public float[] Flush()
    {
        var y = psola != null ? psola.Flush() : Array.Empty<float>();
        if (warp == null)
        {
            return y;
        }
        var head = warp.Push(y);
        var tail = warp.Flush();
        return Samples.Concat(head, tail);
    }
}

/// <summary>Mutlak indeksli, baştan budanabilen örnek tamponu.</summary>
internal sealed class Samples
{
    float[] data = new float[4096];
    int count;

    public long Start { get; private set; }
    public long End => Start + count;

    public float this[long i] => i < Start || i >= End ? 0f : data[i - Start];

    public void Append(ReadOnlySpan<float> x)
    {
        Reserve(count + x.Length);
        x.CopyTo(data.AsSpan(count));
        count += x.Length;
    }

    /// <summary>OLA çıktısı için: indekse ekler, gerekirse sıfırla uzatır; Start'tan önceki indeksler yok sayılır.</summary>
    public void Add(long i, float v)
    {
        if (i < Start)
        {
            return;
        }
        var offset = (int)(i - Start);
        if (offset >= count)
        {
            Reserve(offset + 1);
            Array.Clear(data, count, offset + 1 - count);
            count = offset + 1;
        }
        data[offset] += v;
    }

    /// <summary>[Start, end) aralığını döndürür ve tampondan atar; eksik kısım sıfırdır.</summary>
    public float[] Take(long end)
    {
        if (end <= Start)
        {
            return Array.Empty<float>();
        }
        var n = (int)(end - Start);
        var output = new float[n];
        Array.Copy(data, 0, output, 0, Math.Min(n, count));
        Drop(end);
        return output;
    }

    public void Drop(long before)
    {
        if (before <= Start)
        {
            return;
        }
        var n = (int)Math.Min(before - Start, count);
        Array.Copy(data, n, data, 0, count - n);
        count -= n;
        Start = before;
    }

    void Reserve(int size)
    {
        if (size > data.Length)
        {
            Array.Resize(ref data, Math.Max(size, data.Length * 2));
        }
    }

    public static float[] Concat(float[] a, float[] b)
    {
        if (a.Length == 0)
        {
            return b;
        }
        if (b.Length == 0)
        {
            return a;
        }
        var result = new float[a.Length + b.Length];
        a.CopyTo(result, 0);
        b.CopyTo(result, a.Length);
        return result;
    }
}

/// <summary>YIN perde takibi + TD-PSOLA; süre değişmez, yalnızca F0 ölçeklenir.</summary>
internal sealed class Psola
{
    const int FrameHop = 240; // 5 ms perde çerçevesi (48 kHz)
    const int Decimation = 6; // perde analizi 8 kHz'te
    const int YinWindow = 320; // 40 ms (8 kHz)
    const int TauMin = 16; // 500 Hz
    const int TauMax = 160; // 50 Hz
    const double YinThreshold = 0.15;
    const double VoicedLimit = 0.35;
    const double SilenceRms = 0.005;
    const int Unvoiced = 480; // sessiz/ötümsüz bölgede 10 ms adım
    const int MaxHalf = TauMax * Decimation; // en uzun tanecik yarı boyu

    readonly double ratio;
    readonly Samples x = new();
    readonly Samples y = new();
    readonly List<float> dec = new(); // 8 kHz özet (baştan tutulur; 32 KB/sn)
    readonly List<double> periods = new(); // çerçeve başına periyot (48 kHz örnek), 0 = ötümsüz
    readonly List<long> marks = new(); // analiz işaretleri
    long inputEnd;
    long nextSynthesis;
    bool flushing;

    public Psola(double ratio)
    {
        this.ratio = ratio;
    }

    public float[] Push(float[] chunk)
    {
        x.Append(chunk);
        inputEnd += chunk.Length;
        return Process();
    }

    public float[] Flush()
    {
        flushing = true;
        var output = Process();
        return Samples.Concat(output, y.Take(inputEnd));
    }

    float[] Process()
    {
        Decimate();
        Track();
        Mark();
        Synthesize();
        // nextSynthesis'ten MaxHalf öncesine artık tanecik eklenmez
        var final = flushing ? inputEnd : Math.Min(nextSynthesis - MaxHalf, inputEnd);
        var output = y.Take(Math.Max(final, y.Start));
        // artık en yakın olamayacak eski işaretleri ve onların tanecik verisini bırak
        while (marks.Count > 2 && marks[1] < nextSynthesis - 2 * MaxHalf)
        {
            marks.RemoveAt(0);
        }
        if (marks.Count > 0)
        {
            x.Drop(marks[0] - MaxHalf - 1);
        }
        return output;
    }

    void Decimate()
    {
        while ((dec.Count + 1L) * Decimation <= inputEnd)
        {
            long s = (long)dec.Count * Decimation;
            var sum = 0f;
            for (var i = 0; i < Decimation; i++)
            {
                sum += x[s + i];
            }
            dec.Add(sum / Decimation);
        }
    }

    void Track()
    {
        while (true)
        {
            var frame = periods.Count;
            var center = frame * (FrameHop / Decimation);
            var start = center - YinWindow / 2;
            if (start + YinWindow + TauMax >= dec.Count &&!(flushing && (long)frame * FrameHop < inputEnd + FrameHop))
            {
                return;
            }
            periods.Add(Yin(start));
        }
    }

    double Yin(int start)
    {
        float S(int i) => i >= 0 && i < dec.Count ? dec[i] : 0f;
        var energy = 0.0;
        for (var j = 0; j < YinWindow; j++)
        {
            energy += S(start + j) * S(start + j);
        }
        if (Math.Sqrt(energy / YinWindow) < SilenceRms)
        {
            return 0;
        }
        var d = new double[TauMax + 2];
        for (var tau = 1; tau <= TauMax + 1; tau++)
        {
            var sum = 0.0;
            for (var j = 0; j < YinWindow; j++)
            {
                var diff = S(start + j) - S(start + j + tau);
                sum += diff * diff;
            }
            d[tau] = sum;
        }
        var cmnd = new double[TauMax + 2];
        cmnd[0] = 1;
        var running = 0.0;
        for (var tau = 1; tau <= TauMax + 1; tau++)
        {
            running += d[tau];
            cmnd[tau] = running > 0 ? d[tau] * tau / running : 1;
        }
        var best = -1;
        for (var tau = TauMin; tau <= TauMax; tau++)
        {
            if (cmnd[tau] < YinThreshold)
            {
                while (tau + 1 <= TauMax && cmnd[tau + 1] < cmnd[tau])
                {
                    tau++;
                }
                best = tau;
                break;
            }
        }
        if (best < 0)
        {
            best = TauMin;
            for (var tau = TauMin; tau <= TauMax; tau++)
            {
                if (cmnd[tau] < cmnd[best])
                {
                    best = tau;
                }
            }
            if (cmnd[best] > VoicedLimit)
            {
                return 0;
            }
        }
        // parabolik ara değer
        var a = cmnd[best - 1];
        var b = cmnd[best];
        var c = cmnd[best + 1];
        var denominator = a - 2 * b + c;
        var shift = Math.Abs(denominator) > 1e-12 ? 0.5 * (a - c) / denominator : 0;
        return (best + Math.Clamp(shift, -0.5, 0.5)) * Decimation;
    }

    /// <summary>t anındaki periyot (48 kHz örnek), 0 = ötümsüz; çerçeve henüz yoksa null.</summary>
    double? Period(long t)
    {
        var frame = (int)Math.Round((double)Math.Max(t, 0) / FrameHop);
        if (frame >= periods.Count)
        {
            return flushing && periods.Count > 0 ? periods[^1] : null;
        }
        return periods[frame];
    }

    void Mark()
    {
        if (marks.Count == 0)
        {
            marks.Add(0);
        }
        while (true)
        {
            var last = marks[^1];
            if (Period(last) is not { } period)
            {
                return;
            }
            var step = period > 0 ? period : Unvoiced;
            var predicted = last + (long)Math.Round(step);
            if (flushing && predicted >= inputEnd + MaxHalf)
            {
                return;
            }
            if (period > 0)
            {
                var reach = (long)(period / 4);
                if (predicted + reach >= inputEnd && !flushing)
                {
                    return;
                }
                if (Period(predicted) is null)
                {
                    return;
                }
                // tahmini konum çevresinde en büyük örneğe hizala (perde senkronu)
                var bestIndex = predicted;
                var bestValue = float.MinValue;
                for (var i = Math.Max(predicted - reach, last + (long)(period / 2)); i <= predicted + reach; i++)
                {
                    if (x[i] > bestValue)
                    {
                        bestValue = x[i];
                        bestIndex = i;
                    }
                }
                marks.Add(bestIndex);
            }
            else
            {
                if (predicted >= inputEnd && !flushing)
                {
                    return;
                }
                marks.Add(predicted);
            }
        }
    }

    void Synthesize()
    {
        while (true)
        {
            var t = nextSynthesis;
            if (flushing ? t >= inputEnd : marks[^1] < t + MaxHalf)
            {
                return;
            }
            if (Period(t) is not { } period)
            {
                return;
            }
            var mark = Nearest(t);
            var markPeriod = Period(mark) ?? 0;
            var half = markPeriod > 0 ? (int)Math.Round(markPeriod) : Unvoiced;
            if (!flushing && mark + half >= inputEnd)
            {
                return;
            }
            for (var i = -half; i <= half; i++)
            {
                var w = 0.5 * (1 + Math.Cos(Math.PI * i / half));
                y.Add(t + i, (float)(x[mark + i] * w));
            }
            var step = period > 0 ? period / ratio : Unvoiced;
            nextSynthesis += Math.Max(1, (long)Math.Round(step));
        }
    }

    long Nearest(long t)
    {
        int lo = 0, hi = marks.Count - 1;
        while (lo < hi)
        {
            var mid = (lo + hi) / 2;
            if (marks[mid] < t)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        if (lo > 0 && t - marks[lo - 1] < marks[lo] - t)
        {
            lo--;
        }
        return marks[lo];
    }
}

/// <summary>LPC zarfını frekans ekseninde ölçekleyerek formantları kaydırır (STFT, Hann, %75 örtüşme).</summary>
internal sealed class FormantWarp
{
    const int N = 2048;
    const int Hop = 512;
    const int Order = 48;
    const double OverlapGain = 1.5; // Hann² toplamı, hop = N/4

    static readonly double[] Window = Enumerable.Range(0, N).Select(n => 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / N)).ToArray();

    readonly double factor;
    readonly Samples x = new();
    readonly Samples y = new();
    long inputEnd;
    long frameStart = -(N - Hop);

    public FormantWarp(double factor)
    {
        this.factor = factor;
    }

    public float[] Push(float[] chunk)
    {
        x.Append(chunk);
        inputEnd += chunk.Length;
        while (frameStart + N <= inputEnd)
        {
            Frame();
        }
        return Emit(Math.Min(frameStart, inputEnd));
    }

    public float[] Flush()
    {
        while (frameStart < inputEnd)
        {
            Frame();
        }
        return Emit(inputEnd);
    }

    float[] Emit(long final)
    {
        var output = y.Take(Math.Max(final, Math.Max(y.Start, 0)));
        x.Drop(frameStart);
        return output;
    }

    void Frame()
    {
        var re = new double[N];
        var im = new double[N];
        for (var n = 0; n < N; n++)
        {
            re[n] = x[frameStart + n] * Window[n];
        }
        var lpc = Lpc(re);
        Fft.Transform(re, im, false);
        if (lpc != null)
        {
            var ar = new double[N];
            var ai = new double[N];
            Array.Copy(lpc, ar, lpc.Length);
            Fft.Transform(ar, ai, false);
            var envelope = new double[N / 2 + 1];
            for (var k = 0; k <= N / 2; k++)
            {
                envelope[k] = 1.0 / Math.Max(Math.Sqrt(ar[k] * ar[k] + ai[k] * ai[k]), 1e-9);
            }
            for (var k = 0; k <= N / 2; k++)
            {
                // E'(k) = E(k / factor): zarf frekans ekseninde ölçeklenir
                var source = Math.Min(k / factor, N / 2);
                var i0 = (int)source;
                var i1 = Math.Min(i0 + 1, N / 2);
                var fraction = source - i0;
                var warped = envelope[i0] * (1 - fraction) + envelope[i1] * fraction;
                var gain = Math.Clamp(warped / envelope[k], 0.05, 20);
                re[k] *= gain;
                im[k] *= gain;
                if (k > 0 && k < N / 2)
                {
                    re[N - k] = re[k];
                    im[N - k] = -im[k];
                }
            }
        }
        Fft.Transform(re, im, true);
        for (var n = 0; n < N; n++)
        {
            y.Add(frameStart + n, (float)(re[n] * Window[n] / OverlapGain));
        }
        frameStart += Hop;
    }

    /// <summary>Otokorelasyon + Levinson-Durbin; sessiz çerçevede null.</summary>
    static double[]? Lpc(double[] frame)
    {
        var r = new double[Order + 1];
        for (var lag = 0; lag <= Order; lag++)
        {
            var sum = 0.0;
            for (var n = lag; n < N; n++)
            {
                sum += frame[n] * frame[n - lag];
            }
            r[lag] = sum;
        }
        if (r[0] < 1e-8)
        {
            return null;
        }
        r[0] *= 1 + 1e-6;
        var a = new double[Order + 1];
        a[0] = 1;
        var error = r[0];
        for (var i = 1; i <= Order; i++)
        {
            var acc = r[i];
            for (var j = 1; j < i; j++)
            {
                acc += a[j] * r[i - j];
            }
            var k = -acc / error;
            var previous = (double[])a.Clone();
            for (var j = 1; j < i; j++)
            {
                a[j] = previous[j] + k * previous[i - j];
            }
            a[i] = k;
            error *= 1 - k * k;
            if (error <= 0)
            {
                return null;
            }
        }
        return a;
    }
}

/// <summary>Yerinde radix-2 karmaşık FFT; inverse = true ise 1/N ölçekli ters dönüşüm.</summary>
internal static class Fft
{
    public static void Transform(double[] re, double[] im, bool inverse)
    {
        var n = re.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            var bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
            {
                j ^= bit;
            }
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }
        for (var length = 2; length <= n; length <<= 1)
        {
            var angle = 2 * Math.PI / length * (inverse ? 1 : -1);
            var wr = Math.Cos(angle);
            var wi = Math.Sin(angle);
            for (var i = 0; i < n; i += length)
            {
                double cr = 1, ci = 0;
                for (var j = 0; j < length / 2; j++)
                {
                    var ur = re[i + j];
                    var ui = im[i + j];
                    var vr = re[i + j + length / 2] * cr - im[i + j + length / 2] * ci;
                    var vi = re[i + j + length / 2] * ci + im[i + j + length / 2] * cr;
                    re[i + j] = ur + vr;
                    im[i + j] = ui + vi;
                    re[i + j + length / 2] = ur - vr;
                    im[i + j + length / 2] = ui - vi;
                    var next = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = next;
                }
            }
        }
        if (inverse)
        {
            for (var i = 0; i < n; i++)
            {
                re[i] /= n;
                im[i] /= n;
            }
        }
    }
}
