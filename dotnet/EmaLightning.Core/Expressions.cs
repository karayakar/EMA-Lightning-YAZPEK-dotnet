// İfade klipleri: metindeki etiket (<laugh>, <laugh_03>, <ayy>) ve emojiler (😂 …) konuşmacının gerçek kayıtlarıyla
// seslendirilir. Kütüphane: <model klasörü>\expressions\<kategori>_NN.wav (kategori adları dosya adlarından).
// Eşlenmeyen emoji / tanınmayan etiket ve kütüphanesi olmayan konuşmacı: sessizce atılır (kullanıcı kararı).
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace EmaLightning;

internal sealed class Expressions
{
    // Kullanıcı onaylı eşleme (2026-10-07)
    static readonly Dictionary<string, string> EmojiMap = new()
    {
        ["😂"] = "laugh", ["🤣"] = "laugh", ["😆"] = "laugh",
        ["😢"] = "cry", ["😭"] = "cry",
        ["😔"] = "sad",
        ["😮‍💨"] = "sigh", ["😌"] = "sigh",
        ["🥱"] = "yawn",
        ["🤧"] = "sniff",
        ["😷"] = "cough",
        ["😩"] = "ufff", ["😫"] = "off",
        ["🙄"] = "puff",
        ["😱"] = "eyvah", ["😨"] = "eyvah",
        ["😬"] = "clear_throat",
    };

    static readonly Regex Tag = new(@"\G<([A-Za-z_]+?)(?:_(\d+))?>"); // \G: yalnız verilen konumda eşleşir
    static readonly Regex Numbered = new(@"^(.*?)_(\d+)$");

    readonly Dictionary<string, List<(string Name, float[] Audio)>> clips = new(StringComparer.OrdinalIgnoreCase);

    Expressions()
    {
    }

    public static Expressions Empty { get; } = new();

    public bool Any => clips.Count > 0;

    public IReadOnlyCollection<string> Names => clips.Keys;

    /// <summary>Klasördeki wav'lar (modelin örnekleme hızında, PCM16 mono) → kategori listesi.</summary>
    public static Expressions Load(string directory, int sampleRate)
    {
        var library = new Expressions();
        if (!Directory.Exists(directory))
        {
            return library;
        }
        foreach (var path in Directory.GetFiles(directory, "*.wav").OrderBy(p => p, StringComparer.Ordinal))
        {
            var (audio, rate) = ReadPcm16(path);
            if (audio == null || rate != sampleRate)
            {
                continue; // farklı hız / format: atlanır
            }
            var stem = Path.GetFileNameWithoutExtension(path);
            var match = Numbered.Match(stem);
            var category = match.Success ? match.Groups[1].Value : stem;
            if (!library.clips.TryGetValue(category, out var list))
            {
                library.clips[category] = list = new List<(string, float[])>();
            }
            list.Add((stem, Prepare(audio, rate)));
        }
        return library;
    }

    /// <summary>Metni sırayla (metin, null) ve (null, klip) parçalarına ayırır; klip seed'le seçilir.</summary>
    public List<(string? Text, float[]? Clip)> Split(string text, long seed)
    {
        var output = new List<(string?, float[]?)>();
        var buffer = new StringBuilder();
        var index = 0;
        var choice = 0;

        void Flush()
        {
            if (buffer.Length > 0)
            {
                output.Add((buffer.ToString(), null));
                buffer.Clear();
            }
        }

        void Emit(string category, int? number)
        {
            var clip = Pick(category, number, seed, choice++);
            if (clip != null)
            {
                Flush();
                output.Add((null, clip));
            }
            else
            {
                buffer.Append(' '); // tanınmayan: sessizce atılır (kelimeler bitişmesin)
            }
        }

        while (index < text.Length)
        {
            var tag = Tag.Match(text, index);
            if (tag.Success && tag.Index == index)
            {
                Emit(tag.Groups[1].Value, tag.Groups[2].Success ? int.Parse(tag.Groups[2].Value, CultureInfo.InvariantCulture) : null);
                index += tag.Length;
                continue;
            }
            var length = StringInfo.GetNextTextElementLength(text.AsSpan(index));
            var grapheme = text.Substring(index, length);
            if (EmojiMap.TryGetValue(grapheme.Replace("️", ""), out var mapped))
            {
                Emit(mapped, null);
            }
            else if (IsEmoji(grapheme))
            {
                buffer.Append(' '); // eşlenmeyen emoji: sessizce atılır
            }
            else
            {
                buffer.Append(grapheme);
            }
            index += length;
        }
        Flush();
        return output;
    }

    float[]? Pick(string category, int? number, long seed, int choice)
    {
        if (!clips.TryGetValue(category, out var list) || list.Count == 0)
        {
            return null;
        }
        if (number is { } n)
        {
            var exact = list.FirstOrDefault(c => Numbered.Match(c.Name) is { Success: true } m &&
                                                 int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture) == n);
            return exact.Audio != null ? (float[])exact.Audio.Clone() : null;
        }
        var random = new Random(unchecked((int)(seed * 31 + choice)));
        return (float[])list[random.Next(list.Count)].Audio.Clone();
    }

    /// <summary>Emoji sayılan grafemler: yaygın emoji blokları ya da VS16/ZWJ içerenler.</summary>
    static bool IsEmoji(string grapheme)
    {
        if (grapheme.Contains('️') || grapheme.Contains('‍'))
        {
            return true;
        }
        var value = char.ConvertToUtf32(grapheme, 0);
        return value is (>= 0x1F000 and <= 0x1FAFF) or (>= 0x2600 and <= 0x27BF) or (>= 0x2B00 and <= 0x2BFF);
    }

    /// <summary>Aktif konuşma RMS'i −23 dBFS (eğitim verisiyle aynı), tepe ≤ 0.95, 10 ms yumuşak giriş/çıkış.</summary>
    static float[] Prepare(float[] audio, int rate)
    {
        var frame = rate / 100;
        var frames = audio.Length / frame;
        var db = new double[Math.Max(frames, 1)];
        for (var f = 0; f < frames; f++)
        {
            var sum = 0.0;
            for (var i = 0; i < frame; i++)
            {
                sum += audio[f * frame + i] * (double)audio[f * frame + i];
            }
            db[f] = 10 * Math.Log10(sum / frame + 1e-12);
        }
        var top = db.Max();
        double energy = 0;
        var count = 0;
        for (var f = 0; f < frames; f++)
        {
            if (db[f] > top - 35)
            {
                for (var i = 0; i < frame; i++)
                {
                    energy += audio[f * frame + i] * (double)audio[f * frame + i];
                }
                count += frame;
            }
        }
        var gain = Math.Pow(10, -23.0 / 20) / Math.Sqrt(energy / Math.Max(count, 1) + 1e-12);
        var peak = audio.Length > 0 ? audio.Max(Math.Abs) * gain : 0;
        if (peak > 0.95)
        {
            gain *= 0.95 / peak;
        }
        var output = audio.Select(x => (float)(x * gain)).ToArray();
        var fade = Math.Min(rate / 100, output.Length / 2);
        for (var i = 0; i < fade; i++)
        {
            var w = (float)i / fade;
            output[i] *= w;
            output[^(i + 1)] *= w;
        }
        return output;
    }

    static (float[]? Audio, int Rate) ReadPcm16(string path)
    {
        using var reader = new BinaryReader(File.OpenRead(path));
        if (new string(reader.ReadChars(4)) != "RIFF")
        {
            return (null, 0);
        }
        reader.ReadInt32();
        if (new string(reader.ReadChars(4)) != "WAVE")
        {
            return (null, 0);
        }
        int rate = 0, channels = 0, bits = 0;
        while (reader.BaseStream.Position + 8 <= reader.BaseStream.Length)
        {
            var id = new string(reader.ReadChars(4));
            var size = reader.ReadInt32();
            if (id == "fmt ")
            {
                var format = reader.ReadInt16();
                channels = reader.ReadInt16();
                rate = reader.ReadInt32();
                reader.ReadInt32();
                reader.ReadInt16();
                bits = reader.ReadInt16();
                reader.BaseStream.Seek(size - 16, SeekOrigin.Current);
                if (format != 1)
                {
                    return (null, 0);
                }
            }
            else if (id == "data")
            {
                if (bits != 16 || channels < 1)
                {
                    return (null, 0);
                }
                var samples = size / 2 / channels;
                var audio = new float[samples];
                for (var i = 0; i < samples; i++)
                {
                    var sum = 0;
                    for (var c = 0; c < channels; c++)
                    {
                        sum += reader.ReadInt16();
                    }
                    audio[i] = sum / (float)channels / 32768f;
                }
                return (audio, rate);
            }
            else
            {
                reader.BaseStream.Seek(size + (size & 1), SeekOrigin.Current);
            }
        }
        return (null, 0);
    }
}
