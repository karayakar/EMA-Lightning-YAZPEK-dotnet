// scheduler.py portu — Playhead: tek arka plan döngüsü, iki FIFO kuyruk, her çağıranın işi ortak batch'lerde.
//   kuyruk 1, modelden önce:   düşünülmeyi bekleyen cümleler
//   kuyruk 2, decoder'dan önce: decode edilmeyi bekleyen pencereler
// Kapanan (bağlantısı kopan) istek iki kuyruktan da düşer; bir aşama hata verirse yalnızca o batch'tekiler hata alır.
using System.Threading.Channels;

namespace EmaLightning;

/// <summary>Bir çağıran: cümleleri ve sırayla ses, sonra Done ya da hata taşıyan outbox.</summary>
internal sealed class Request
{
    public static readonly object Done = new();

    /// <summary>Bir cümlenin son penceresinden ve duraklamasından sonra gelir; cümle bazlı ses dönüştürme için.</summary>
    public static readonly object PieceEnd = new();

    volatile bool closed;

    public Request(List<Piece> pieces, double speed, int first, int rate)
    {
        Pieces = pieces;
        Speed = speed;
        First = first;
        Rate = rate;
    }

    /// <summary>Modelin kendi örnekleme hızı (duraklama sessizliği bu hızda üretilir).</summary>
    public int Rate { get; }

    public List<Piece> Pieces { get; }
    public double Speed { get; }
    public int First { get; }
    public Channel<object> Outbox { get; } = Channel.CreateUnbounded<object>();
    public bool Closed => closed;
    public int Left { get; set; } // teslim edilecek pencere sayısı

    public void Cancel() => closed = true;

    public void Finish(Exception? error = null)
    {
        if (!closed)
        {
            closed = true;
            Outbox.Writer.TryWrite(error ?? Done);
        }
    }

    public void Send(Piece piece, bool last, float[] audio)
    {
        Outbox.Writer.TryWrite(audio);
        if (last)
        {
            if (piece.Pause != 0)
            {
                Outbox.Writer.TryWrite(new float[(int)Math.Round(piece.Pause * Rate)]);
            }
            Outbox.Writer.TryWrite(PieceEnd);
            piece.H = piece.Dur = piece.Latents = null;
        }
        Left--;
        if (Left == 0)
        {
            Finish();
        }
    }
}

internal sealed class Playhead
{
    readonly Engine engine;
    readonly Func<int> batchSize;
    readonly object cond = new();
    List<Request> inbox = new();
    List<(Request Request, Piece Piece)> sentences = new(); // kuyruk 1
    List<(Request Request, Piece Piece, (int Start, int End) Span, bool Last)> windows = new(); // kuyruk 2
    Thread? thread;

    public Playhead(Engine engine, Func<int> batchSize)
    {
        this.engine = engine;
        this.batchSize = batchSize;
    }

    /// <summary>
    /// Queue one caller's sentences; returns the request whose outbox carries their audio.
    /// `first`: ilk cümlenin ilk penceresi (kare): say() için 4 sn, stream() için 1 sn.
    /// </summary>
    public Request Submit(List<Piece> pieces, double speed, int first = Engine.Window)
    {
        var request = new Request(pieces, speed, first, engine.SampleRate);
        if (pieces.Count == 0)
        {
            request.Finish();
            return request;
        }
        lock (cond)
        {
            inbox.Add(request);
            if (thread == null)
            {
                thread = new Thread(Loop) { Name = "ema-playhead", IsBackground = true };
                thread.Start();
            }
            Monitor.Pulse(cond);
        }
        return request;
    }

    void Loop()
    {
        while (true)
        {
            List<Request> arrived;
            lock (cond)
            {
                while (inbox.Count == 0 && sentences.Count == 0 && windows.Count == 0)
                {
                    Monitor.Wait(cond);
                }
                arrived = inbox;
                inbox = new List<Request>();
            }
            try
            {
                lock (engine.Lock)
                {
                    Turn(arrived);
                }
            }
            catch (Exception error)
            {
                // aşamaların dışındaki her şey: uçuştaki herkesi hata ile bitir, hizmete devam et
                var callers = new List<Request>(arrived);
                callers.AddRange(sentences.Select(x => x.Request));
                callers.AddRange(windows.Select(x => x.Request));
                Fail(callers.Distinct(), error);
            }
        }
    }

    void Turn(List<Request> arrived)
    {
        var size = Math.Max(1, batchSize());
        PlanNew(arrived.Where(r => !r.Closed).ToList(), size);
        DropClosed();
        Think(size);
        Decode(size);
    }

    /// <summary>Plan every newcomer's sentences, then put them at the back of queue 1 in arrival order.</summary>
    void PlanNew(List<Request> requests, int size)
    {
        foreach (var group in requests.GroupBy(r => r.Speed))
        {
            var owner = new Dictionary<Piece, Request>(ReferenceEqualityComparer.Instance);
            foreach (var r in group)
            {
                foreach (var p in r.Pieces)
                {
                    owner[p] = r;
                }
            }
            // ifade klipleri modelden geçmez
            var pieces = group.SelectMany(r => r.Pieces).Where(p => p.Clip == null).OrderBy(p => p.Letters).ToList();
            foreach (var batch in pieces.Chunk(size))
            {
                try
                {
                    engine.Plan(batch, group.Key);
                }
                catch (Exception error)
                {
                    Fail(batch.Select(p => owner[p]).Distinct(), error);
                }
            }
        }
        foreach (var r in requests)
        {
            if (r.Closed)
            {
                continue;
            }
            for (var i = 0; i < r.Pieces.Count; i++)
            {
                var p = r.Pieces[i];
                p.Spans = p.Clip != null ? new List<(int, int)> { (0, 0) } : Engine.Windows(p.Frames, i == 0 ? r.First : Engine.Window);
            }
            r.Left = r.Pieces.Sum(p => p.Spans!.Count);
            sentences.AddRange(r.Pieces.Select(p => (r, p)));
        }
    }

    /// <summary>Think the batch at the front of queue 1; their windows go to the back of queue 2.</summary>
    void Think(int size)
    {
        var count = Math.Min(size, sentences.Count);
        if (count == 0)
        {
            return;
        }
        // Stream başı (Solo) tek başına düşünülür; batch bir Solo parçaya gelince orada kesilir (sıra bozulmaz)
        if (sentences[0].Piece.Solo)
        {
            count = 1;
        }
        else
        {
            for (var i = 1; i < count; i++)
            {
                if (sentences[i].Piece.Solo)
                {
                    count = i;
                    break;
                }
            }
        }
        var batch = sentences.GetRange(0, count);
        sentences.RemoveRange(0, count);
        try
        {
            var model = batch.Select(x => x.Piece).Where(p => p.Clip == null).ToList();
            if (model.Count > 0)
            {
                engine.Think(model);
            }
        }
        catch (Exception error)
        {
            Fail(batch.Select(x => x.Request).Distinct(), error);
            return;
        }
        foreach (var (r, p) in batch)
        {
            if (!r.Closed)
            {
                for (var i = 0; i < p.Spans!.Count; i++)
                {
                    windows.Add((r, p, p.Spans[i], i == p.Spans.Count - 1));
                }
            }
        }
    }

    /// <summary>Decode the batch at the front of queue 2 and hand each window to its caller.</summary>
    void Decode(int size)
    {
        // CPU'da sabit decode boyu yok (engine.decode_sizes boş), bu yüzden boya göre bölme yapılmaz
        var count = Math.Min(size, windows.Count);
        if (count == 0)
        {
            return;
        }
        var batch = windows.GetRange(0, count);
        windows.RemoveRange(0, count);
        List<float[]> audio;
        try
        {
            // ifade klipleri decode edilmez, sıradaki yerlerine hazır sesleri konur
            var model = batch.Where(x => x.Piece.Clip == null).Select(x => (x.Piece, x.Span)).ToList();
            var decoded = new Queue<float[]>(model.Count > 0 ? engine.Decode(model) : new List<float[]>());
            audio = batch.Select(x => x.Piece.Clip != null ? (float[])x.Piece.Clip.Clone() : decoded.Dequeue()).ToList();
        }
        catch (Exception error)
        {
            Fail(batch.Select(x => x.Request).Distinct(), error);
            return;
        }
        for (var i = 0; i < batch.Count; i++)
        {
            var (r, p, _, last) = batch[i];
            if (!r.Closed)
            {
                r.Send(p, last, audio[i]);
            }
        }
    }

    void Fail(IEnumerable<Request> requests, Exception error)
    {
        foreach (var r in requests)
        {
            r.Finish(error);
        }
        DropClosed();
    }

    void DropClosed()
    {
        sentences = sentences.Where(x => !x.Request.Closed).ToList();
        windows = windows.Where(x => !x.Request.Closed).ToList();
    }
}
