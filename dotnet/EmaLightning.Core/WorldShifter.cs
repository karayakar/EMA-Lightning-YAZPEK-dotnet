// A yolu: WORLD (native world_shift.dll) ile cümle bazında F0 ve formant dönüştürme.
using System.Runtime.InteropServices;

namespace EmaLightning;

/// <summary>Ses dönüştürme yöntemi.</summary>
public enum VoiceMethod
{
    /// <summary>WORLD, F0 analizi Harvest (daha doğru, daha yavaş).</summary>
    Harvest,

    /// <summary>WORLD, F0 analizi DIO + StoneMask (daha hızlı).</summary>
    Dio,

    /// <summary>B yolu: YIN + TD-PSOLA + LPC zarf bükme (akış halinde, bağımlılıksız).</summary>
    Psola,
}

internal static class WorldShifter
{
    const double FramePeriod = 5.0; // ms

    [DllImport("world_shift", EntryPoint = "world_shift", CallingConvention = CallingConvention.Cdecl)]
    static extern int Shift([In] double[] x, int length, int fs, double pitch, double formant, int f0Method,
        double framePeriod, [Out] double[] y);

    /// <summary>Bir cümlenin tamamını dönüştürür; çıkış aynı uzunluktadır.</summary>
    public static float[] Process(float[] sentence, double pitch, double formant, VoiceMethod method)
    {
        if (sentence.Length == 0)
        {
            return sentence;
        }
        var x = Array.ConvertAll(sentence, v => (double)v);
        var y = new double[x.Length];
        if (Shift(x, x.Length, Engine.Rate, pitch, formant, method == VoiceMethod.Harvest ? 0 : 1, FramePeriod, y) != 1)
        {
            throw new InvalidOperationException("world_shift failed");
        }
        return Array.ConvertAll(y, v => (float)v);
    }
}
