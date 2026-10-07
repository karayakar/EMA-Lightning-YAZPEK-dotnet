"""HF parquet (text + audio{bytes, path}) → wavs\\ + metadata.csv (100HRSKARAY düzeni: "ad.wav| metin").

Ses baytları olduğu gibi yazılır (yeniden kodlama yok). Parquet dosyalarına dokunulmaz.
  .venv\\Scripts\\python.exe extract_parquet.py --dataset G:\\OPENAI\\Turkish_VoiceDatasets\\tts_karay2_tur
Çıktı: <dataset>\\wavs\\, <dataset>\\metadata.csv, <dataset>\\extract_report.json
"""
import argparse
import io
import json
import wave
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = Path(args.dataset)
    wavs = root / "wavs"
    wavs.mkdir(exist_ok=True)
    formats, problems, names = Counter(), [], set()
    seconds = 0.0
    rows = 0
    with open(root / "metadata.csv", "w", encoding="utf-8", newline="\n") as meta:
        for part in sorted((root / "data").glob("*.parquet")):
            table = pq.ParquetFile(part)
            for group in range(table.num_row_groups):
                for row in table.read_row_group(group).to_pylist():
                    rows += 1
                    audio = row["audio"] or {}
                    data = audio.get("bytes") or b""
                    name = Path(audio.get("path") or f"row_{rows}.wav").name
                    if name in names:  # aynı ad: satır numarası eklenir
                        name = f"{Path(name).stem}__{rows}.wav"
                    names.add(name)
                    if not data:
                        problems.append({"row": rows, "name": name, "reason": "no_audio"})
                        continue
                    try:
                        with wave.open(io.BytesIO(data)) as w:
                            formats[(w.getframerate(), w.getsampwidth() * 8, w.getnchannels())] += 1
                            seconds += w.getnframes() / w.getframerate()
                    except Exception as error:  # wav değilse yine yazılır, raporlanır
                        problems.append({"row": rows, "name": name, "reason": f"not_wav: {error}"})
                    (wavs / name).write_bytes(data)
                    text = (row.get("text") or "").replace("\r", " ").replace("\n", " ").strip()
                    meta.write(f"{name}| {text}\n")
            print(f"{part.name}: toplam {rows} satır, {seconds / 3600:.2f} saat", flush=True)
    report = {"rows": rows, "hours": round(seconds / 3600, 2),
              "formats (Hz, bit, kanal)": {str(k): v for k, v in formats.items()}, "problems": problems[:200],
              "problem_count": len(problems)}
    (root / "extract_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "problems"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
