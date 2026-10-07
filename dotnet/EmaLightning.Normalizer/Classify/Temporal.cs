// src/classify/temporal.rs portu
namespace NormalizerTr;

internal static class Temporal
{
    public static Res<Value, TemporalFailure> DateReading(string text, bool permitted)
    {
        if (!NumericText.SplitSuffix(text, out var @base, out var suffix) || Date.Parse(@base) is not { } date)
        {
            return Res<Value, TemporalFailure>.Err(new(IssueCategory.InvalidExpression));
        }
        if (suffix != null && !date.Dotted)
        {
            return Res<Value, TemporalFailure>.Err(new(IssueCategory.Unsupported));
        }
        if (suffix != null && Verbalize.DateSpoken(date).SourceSuffix(Inflection.Locative) != suffix)
        {
            return Res<Value, TemporalFailure>.Err(new(IssueCategory.InvalidExpression));
        }
        if (!permitted)
        {
            return Res<Value, TemporalFailure>.Err(new(IssueCategory.Ambiguous,
                new DatePreference(date, suffix != null)));
        }
        return Res<Value, TemporalFailure>.Ok(new DateValue(date, suffix != null));
    }

    public static Res<Value, TemporalFailure> TimeReading(string text, bool permitted)
    {
        if (!NumericText.SplitSuffix(text, out var @base, out var suffix) || Clock.Parse(@base) is not { } time)
        {
            return Res<Value, TemporalFailure>.Err(new(IssueCategory.InvalidExpression));
        }
        if (suffix != null && Verbalize.TimeSpoken(time).SourceSuffix(Inflection.Locative) != suffix)
        {
            return Res<Value, TemporalFailure>.Err(new(IssueCategory.InvalidExpression));
        }
        if (!permitted)
        {
            return Res<Value, TemporalFailure>.Err(new(IssueCategory.Ambiguous,
                new TimePreference(time, suffix != null)));
        }
        return Res<Value, TemporalFailure>.Ok(new TimeValue(time, suffix != null));
    }

    static bool Cue(string text, List<Token> tokens, int index, string[] allowed)
    {
        var previousIndex = index - 1;
        if (previousIndex < 0)
        {
            return false;
        }
        bool Adjacent(int start, int end) => start == end || Scan.WhitespaceBetween(text, start, end);
        if (!Adjacent(tokens[previousIndex].Range.End, tokens[index].Range.Start))
        {
            return false;
        }
        if (tokens[previousIndex].Text == ":")
        {
            var wordIndex = previousIndex - 1;
            if (wordIndex < 0)
            {
                return false;
            }
            if (!Scan.WhitespaceBetween(text, tokens[wordIndex].Range.End, tokens[previousIndex].Range.Start)
                || tokens[wordIndex].Text.Contains(':'))
            {
                return false;
            }
            previousIndex = wordIndex;
        }
        return allowed.Contains(CueContext.CueKey(tokens[previousIndex].Text));
    }

    static bool Frame(string text, List<Token> tokens, int index)
    {
        if (index + 2 >= tokens.Count)
        {
            return false;
        }
        var dateToken = tokens[index];
        var cueToken = tokens[index + 1];
        var timeToken = tokens[index + 2];
        var date = DateReading(dateToken.Text, true);
        var time = TimeReading(timeToken.Text, true);
        return date.IsOk && date.Value is DateValue { Locative: true } d && d.Date.Dotted
               && CueContext.IsClockWord(cueToken.Text)
               && time.IsOk && time.Value is TimeValue { Locative: true }
               && Scan.WhitespaceBetween(text, dateToken.Range.End, cueToken.Range.Start)
               && Scan.WhitespaceBetween(text, cueToken.Range.End, timeToken.Range.Start);
    }

    public static (Res<Value, TemporalFailure> Reading, FallbackClass Class)? Recognize(string text,
        List<Token> tokens, int index)
    {
        var token = tokens[index].Text;
        if (!NumericText.SplitSuffix(token, out var @base, out _))
        {
            return null;
        }
        var dots = Uni.Count(@base, '.');
        var hyphens = Uni.Count(@base, '-');
        if (dots == 2 || (hyphens == 2 && Uni.Len8(@base) >= 8))
        {
            if (dots == 2
                && Number.Parse(@base) is { Grouped: true }
                && !Cue(text, tokens, index, CueContext.DateCues))
            {
                return null;
            }
            return (DateReading(token, Frame(text, tokens, index) || Cue(text, tokens, index, CueContext.DateCues)),
                FallbackClass.Date);
        }
        if (@base.Contains(':') || (dots == 1 && Cue(text, tokens, index, CueContext.TimeCues)))
        {
            return (TimeReading(token, Cue(text, tokens, index, CueContext.TimeCues)), FallbackClass.Time);
        }
        return null;
    }
}
