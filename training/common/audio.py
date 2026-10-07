"""Ses yardımcıları: PCM16 wav okuma/yazma, sessizlik kırpma, seviye eşitleme, bant sınırı ölçümü."""
import math
import wave

import numpy as np

RATE = 24000  # erkek modeli 24 kHz'te eğitilir (TRAINING_PROPOSAL §8)


def read_wav(path):
    """PCM16 wav → (float32 [-1, 1] mono, örnekleme hızı, kanal sayısı)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"PCM16 bekleniyordu ({w.getsampwidth() * 8} bit)")
        rate, channels = w.getframerate(), w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate, channels


def write_wav(path, audio, rate=RATE):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def frame_db(audio, frame):
    """10 ms'lik çerçeve başına RMS (dB, mutlak)."""
    n = len(audio) // frame
    if n == 0:
        return np.full(1, -120.0)
    frames = audio[: n * frame].reshape(n, frame).astype(np.float64)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1) + 1e-12))


def trim(audio, rate, threshold_db=-45.0, pad_seconds=0.1):
    """Baş ve sondaki sessizlik: en yüksek çerçevenin threshold_db altı; (başlangıç, bitiş) örnek. Sessizse (0, 0)."""
    frame = rate // 100
    db = frame_db(audio, frame)
    active = np.where(db > max(db.max() + threshold_db, -80.0))[0]
    if len(active) == 0:
        return 0, 0
    pad = int(pad_seconds * rate)
    return max(0, int(active[0]) * frame - pad), min(len(audio), (int(active[-1]) + 1) * frame + pad)


def gain_for(audio, rate, target_db=-23.0, active_db=-35.0, peak=0.95):
    """Aktif konuşma RMS'ini target_db'ye getiren, tepe değeri peak ile sınırlanmış kazanç."""
    frame = rate // 100
    db = frame_db(audio, frame)
    n = len(db) * frame
    if n == 0:
        return 1.0
    frames = audio[:n].reshape(len(db), frame).astype(np.float64)
    active = frames[db > db.max() + active_db]
    rms = math.sqrt(float((active ** 2).mean()) + 1e-12)
    gain = 10 ** (target_db / 20) / rms
    top = float(np.abs(audio).max()) * gain
    return gain * (peak / top) if top > peak else gain


def band_edge(audio, rate, floor_db=-60.0, n_fft=2048):
    """Ortalama güç spektrumunun tepeye göre floor_db üstünde kaldığı en yüksek frekans (Hz)."""
    m = len(audio) // n_fft
    if m == 0:
        return 0.0
    frames = audio[: m * n_fft].reshape(m, n_fft).astype(np.float64) * np.hanning(n_fft)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    db = 10 * np.log10(power / power.max() + 1e-20)
    return float(np.fft.rfftfreq(n_fft, 1 / rate)[np.where(db > floor_db)[0][-1]])
