// chunker.py portu: konuşma metni → modelin tek geçişte okuyacağı parçalar (her biri ~10 sn'yi aşmaz).
using System.Text.RegularExpressions;

namespace EmaLightning;

internal static class Chunker
{
    const double LettersPerSecond = 18.0;
    const double MaxSeconds = 10.0;
    const int MaxLetters = 250;
    const double SentencePause = 0.25;
    const double ClausePause = 0.12;

    static readonly (Regex Pattern, double Gap)[] Cuts =
    {
        (new Regex(@"[.!?]+[""')]*(?= )"), SentencePause),
        (new Regex(@"[,;:](?= )"), ClausePause),
        (new Regex(@"\S(?= )"), ClausePause),
    };

    static readonly Regex Letter = new(@"[^\W\d_]");

    /// <summary>[(parça, ardından gelecek sessizlik sn)], her parça cümle sonu noktalamasıyla biter.</summary>
    public static List<(string Piece, double Pause)> Chunk(string text, double speed)
    {
        var limit = (int)Math.Min(MaxLetters, LettersPerSecond * MaxSeconds * speed);
        var pieces = new List<(string, double)>();
        var rest = text.Trim();
        while (rest.Length > 0)
        {
            var cut = rest.Length;
            var pause = 0.0;
            if (rest.Length > limit)
            {
                cut = limit;
                // Python finditer(rest, 0, limit + 1): bakış (lookahead) endpos'un ötesini görmez
                var window = rest[..Math.Min(rest.Length, limit + 1)];
                foreach (var (pattern, gap) in Cuts)
                {
                    var matches = pattern.Matches(window);
                    if (matches.Count > 0)
                    {
                        var last = matches[^1];
                        cut = last.Index + last.Length;
                        pause = gap;
                        break;
                    }
                }
            }
            var piece = rest[..cut].Trim();
            rest = rest[cut..].Trim();
            if (Letter.IsMatch(piece))
            {
                pieces.Add((Finish(piece), pause));
            }
        }
        if (pieces.Count > 0)
        {
            pieces[^1] = (pieces[^1].Item1, 0.0);
        }
        return pieces;
    }

    /// <summary>End every piece the way the model was trained: on a sentence end.</summary>
    internal static string Finish(string piece)
    {
        var stripped = piece.TrimEnd('"', '\'', ')');
        if (stripped.Length > 0 && stripped[^1] is '.' or '!' or '?')
        {
            return piece;
        }
        return piece.TrimEnd(',', ';', ':', '-', ' ') + ".";
    }
}
