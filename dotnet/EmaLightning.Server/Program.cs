// EMA Lightning HTTP/WebSocket servisi: konuşmacı (model seti) başına bir Ema, bağlantı başına bir istek.
//   GET  /voices: yüklü konuşmacılar
//   POST /tts   : JSON {text, speaker?, speed?, seed?, sample_rate?, pitch?, formant?, method?} → audio/wav (16-bit PCM)
//                 speaker: appsettings Ema:Speakers anahtarları (varsayılan Ema:DefaultSpeaker)
//                 method: harvest (WORLD, varsayılan) | dio (WORLD) | psola
//   WS   /speak : ilk mesaj aynı JSON; sunucu float32 PCM (little-endian, mono) binary parçalar yollar, bitince kapatır
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using EmaLightning;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddCors(options =>
    options.AddDefaultPolicy(policy => policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod()));
// Ema:Speakers { ad: klasör }; klasörde ema_meta.json yoksa o konuşmacı atlanır (ör. erkek modeli henüz yok)
var speakerDirectories = builder.Configuration.GetSection("Ema:Speakers").GetChildren()
    .ToDictionary(s => s.Key, s => Path.GetFullPath(Path.Combine(builder.Environment.ContentRootPath, s.Value!)));
builder.Services.AddSingleton(_ => new Speakers(
    speakerDirectories.Where(s => File.Exists(Path.Combine(s.Value, "ema_meta.json")))
        .ToDictionary(s => s.Key, s => new Ema(s.Value)),
    builder.Configuration["Ema:DefaultSpeaker"] ?? "female"));

var app = builder.Build();
app.UseCors();
app.UseDefaultFiles(); // GET / → wwwroot/index.html (test sayfası)
app.UseStaticFiles();
app.UseWebSockets();

// Açılışta ölç (cache varsa anında döner); ilk istek ölçümü beklemez
var speakers = app.Services.GetRequiredService<Speakers>();
foreach (var (name, model) in speakers.All)
{
    app.Logger.LogInformation("EMA Lightning ready: speaker {Speaker} ({Directory}), batch size {BatchSize}", name,
        speakerDirectories[name], model.BestBatchSize());
}

app.MapGet("/voices", (Speakers s) => Results.Ok(new { speakers = s.All.Keys, @default = s.Default }));

app.MapPost("/tts", async (TtsRequest request, Speakers speakers, CancellationToken cancellationToken) =>
{
    if (request.Text == null)
    {
        return Results.BadRequest(new { error = "text is required" });
    }
    try
    {
        var speech = await speakers.Get(request.Speaker).SayAsync(request.Text, request.Speed ?? 1.0, request.Seed,
            request.SampleRate ?? 48000, request.Pitch ?? 1.0, request.Formant ?? 1.0, ParseMethod(request.Method),
            cancellationToken: cancellationToken);
        var wav = new MemoryStream();
        Wav.Write(wav, speech.Audio, speech.SampleRate);
        wav.Position = 0;
        return Results.File(wav, "audio/wav");
    }
    catch (ArgumentException error)
    {
        return Results.BadRequest(new { error = error.Message });
    }
});

app.Map("/speak", async (HttpContext context, Speakers speakers) =>
{
    if (!context.WebSockets.IsWebSocketRequest)
    {
        context.Response.StatusCode = StatusCodes.Status400BadRequest;
        return;
    }
    using var socket = await context.WebSockets.AcceptWebSocketAsync();
    var cancellationToken = context.RequestAborted;
    var message = await ReceiveText(socket, cancellationToken);
    if (message == null)
    {
        return;
    }
    TtsRequest? request;
    try
    {
        request = JsonSerializer.Deserialize<TtsRequest>(message);
    }
    catch (JsonException)
    {
        request = null;
    }
    if (request?.Text == null)
    {
        await socket.CloseAsync(WebSocketCloseStatus.InvalidPayloadData, "expected JSON with text",
            cancellationToken);
        return;
    }
    try
    {
        await foreach (var chunk in speakers.Get(request.Speaker).StreamAsync(request.Text, request.Speed ?? 1.0, request.Seed,
                           request.SampleRate ?? 48000, request.Pitch ?? 1.0, request.Formant ?? 1.0,
                           ParseMethod(request.Method), cancellationToken))
        {
            var bytes = new byte[chunk.Length * sizeof(float)];
            Buffer.BlockCopy(chunk, 0, bytes, 0, bytes.Length);
            await socket.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Binary, true, cancellationToken);
        }
        await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", cancellationToken);
    }
    catch (ArgumentException error)
    {
        await socket.CloseAsync(WebSocketCloseStatus.InvalidPayloadData, error.Message, cancellationToken);
    }
    catch (OperationCanceledException)
    {
        // istemci koptu; Ema o metnin kalan işini düşürdü
    }
    catch (Exception error) when (socket.State == WebSocketState.Open)
    {
        app.Logger.LogError(error, "speech failed");
        await socket.CloseAsync(WebSocketCloseStatus.InternalServerError, "speech failed", CancellationToken.None);
    }
});

app.Run();

// "harvest" (varsayılan) | "dio" | "psola"
static VoiceMethod ParseMethod(string? method) => method?.ToLowerInvariant() switch
{
    null or "" or "harvest" => VoiceMethod.Harvest,
    "dio" => VoiceMethod.Dio,
    "psola" => VoiceMethod.Psola,
    _ => throw new ArgumentException("method must be harvest, dio or psola", nameof(method)),
};

static async Task<string?> ReceiveText(WebSocket socket, CancellationToken cancellationToken)
{
    var buffer = new byte[4096];
    using var message = new MemoryStream();
    while (true)
    {
        var result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationToken);
        if (result.MessageType == WebSocketMessageType.Close)
        {
            return null;
        }
        message.Write(buffer, 0, result.Count);
        if (result.EndOfMessage)
        {
            return Encoding.UTF8.GetString(message.ToArray());
        }
    }
}

record TtsRequest(
    [property: JsonPropertyName("text")] string? Text,
    [property: JsonPropertyName("speed")] double? Speed,
    [property: JsonPropertyName("seed")] long? Seed,
    [property: JsonPropertyName("sample_rate")] int? SampleRate,
    [property: JsonPropertyName("pitch")] double? Pitch,
    [property: JsonPropertyName("formant")] double? Formant,
    [property: JsonPropertyName("method")] string? Method,
    [property: JsonPropertyName("speaker")] string? Speaker);

/// <summary>Konuşmacı adı → Ema (model seti).</summary>
sealed class Speakers(Dictionary<string, Ema> all, string defaultSpeaker)
{
    public IReadOnlyDictionary<string, Ema> All => all;
    public string Default => all.ContainsKey(defaultSpeaker) ? defaultSpeaker : all.Keys.First();

    public Ema Get(string? name) => all.TryGetValue(string.IsNullOrEmpty(name) ? Default : name, out var ema)
        ? ema
        : throw new ArgumentException($"speaker must be one of {string.Join(", ", all.Keys)}", "speaker");
}
