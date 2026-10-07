// EMA Lightning için WORLD tabanlı ses dönüştürme (A yolu).
// Tek çağrı: analiz (Harvest ya da DIO + StoneMask, CheapTrick, D4C) → F0 × pitch,
// spektral zarf ve aperiodiklik frekans ekseninde × formant → sentez. Çıkış girişle aynı uzunlukta.
// WORLD: Copyright (c) 2010 M. Morise, modified BSD (github\World\LICENSE.txt).
#include <algorithm>
#include <vector>

#include "world/cheaptrick.h"
#include "world/d4c.h"
#include "world/dio.h"
#include "world/harvest.h"
#include "world/stonemask.h"
#include "world/synthesis.h"

namespace {

// out[k] = in[k / factor] (doğrusal ara değer, Nyquist'te kırpılır)
void Warp(std::vector<double>& row, double factor) {
  const int bins = static_cast<int>(row.size());
  std::vector<double> source(row);
  for (int k = 0; k < bins; ++k) {
    const double position = std::min(k / factor, static_cast<double>(bins - 1));
    const int i0 = static_cast<int>(position);
    const int i1 = std::min(i0 + 1, bins - 1);
    const double fraction = position - i0;
    row[k] = source[i0] * (1.0 - fraction) + source[i1] * fraction;
  }
}

}  // namespace

// f0_method: 0 = Harvest, 1 = DIO + StoneMask. Başarıda 1, geçersiz girdide 0 döner.
extern "C" __declspec(dllexport) int world_shift(const double* x, int length, int fs, double pitch,
                                                 double formant, int f0_method, double frame_period,
                                                 double* y) {
  if (x == nullptr || y == nullptr || length <= 0 || fs <= 0 || pitch <= 0 || formant <= 0) {
    return 0;
  }
  int frames;
  std::vector<double> times;
  std::vector<double> f0;
  if (f0_method == 0) {
    HarvestOption option;
    InitializeHarvestOption(&option);
    option.frame_period = frame_period;
    frames = GetSamplesForHarvest(fs, length, frame_period);
    times.resize(frames);
    f0.resize(frames);
    Harvest(x, length, fs, &option, times.data(), f0.data());
  } else {
    DioOption option;
    InitializeDioOption(&option);
    option.frame_period = frame_period;
    frames = GetSamplesForDIO(fs, length, frame_period);
    times.resize(frames);
    std::vector<double> raw(frames);
    Dio(x, length, fs, &option, times.data(), raw.data());
    f0.resize(frames);
    StoneMask(x, length, fs, times.data(), raw.data(), frames, f0.data());
  }

  CheapTrickOption cheaptrick;
  InitializeCheapTrickOption(fs, &cheaptrick);
  const int fft_size = cheaptrick.fft_size;
  const int bins = fft_size / 2 + 1;
  std::vector<std::vector<double>> spectrogram(frames, std::vector<double>(bins));
  std::vector<std::vector<double>> aperiodicity(frames, std::vector<double>(bins));
  std::vector<double*> sp(frames);
  std::vector<double*> ap(frames);
  for (int i = 0; i < frames; ++i) {
    sp[i] = spectrogram[i].data();
    ap[i] = aperiodicity[i].data();
  }
  CheapTrick(x, length, fs, times.data(), f0.data(), frames, &cheaptrick, sp.data());
  D4COption d4c;
  InitializeD4COption(&d4c);
  D4C(x, length, fs, times.data(), f0.data(), frames, fft_size, &d4c, ap.data());

  for (int i = 0; i < frames; ++i) {
    f0[i] *= pitch;
    if (formant != 1.0) {
      Warp(spectrogram[i], formant);
      Warp(aperiodicity[i], formant);
    }
  }
  Synthesis(f0.data(), frames, sp.data(), ap.data(), fft_size, frame_period, fs, length, y);
  return 1;
}
