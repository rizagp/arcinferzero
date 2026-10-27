// Audio input: decoding to 16 kHz mono float samples. WAV (PCM 8/16/24/32-bit or float32, any rate and channel
// count) is parsed here; other containers (mp3, m4a/aac, flac, wma, ...) go through Windows Media Foundation.
// Resampling: windowed-sinc low-pass interpolation (Blackman window).
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <shlwapi.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "ole32.lib")

namespace audio {

// RIFF/WAVE: returns false when the bytes are not a WAV file; throws on an unsupported WAV encoding.
inline bool parse_wav(const std::vector<uint8_t>& b, std::vector<float>& mono, int& rate) {
  auto u16 = [&](size_t o) { return (uint32_t)b[o] | (uint32_t)b[o + 1] << 8; };
  auto u32 = [&](size_t o) { return u16(o) | u16(o + 2) << 16; };
  if (b.size() < 12 || memcmp(b.data(), "RIFF", 4) || memcmp(b.data() + 8, "WAVE", 4)) return false;
  int fmt = 0, ch = 0, bits = 0;
  size_t p = 12;
  while (p + 8 <= b.size()) {
    const uint32_t len = u32(p + 4);
    const size_t body = p + 8, avail = std::min<size_t>(len, b.size() - body);
    if (!memcmp(b.data() + p, "fmt ", 4) && avail >= 16) {
      fmt = (int)u16(body);
      ch = (int)u16(body + 2);
      rate = (int)u32(body + 4);
      bits = (int)u16(body + 14);
      if (fmt == 0xFFFE && avail >= 26) fmt = (int)u16(body + 24);  // WAVE_FORMAT_EXTENSIBLE: sub-format GUID
    } else if (!memcmp(b.data() + p, "data", 4)) {
      if (ch <= 0 || rate <= 0) throw std::runtime_error("wav: data before fmt");
      const int bps = bits / 8;
      if (!((fmt == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (fmt == 3 && bits == 32)))
        throw std::runtime_error("wav: unsupported encoding (PCM 8/16/24/32-bit or float32 expected)");
      const size_t n = avail / ((size_t)bps * ch);
      mono.assign(n, 0.0f);
      const uint8_t* d = b.data() + body;
      for (size_t i = 0; i < n; ++i) {
        float acc = 0;
        for (int c = 0; c < ch; ++c) {
          const uint8_t* s = d + (i * ch + c) * bps;
          float v;
          if (fmt == 3) memcpy(&v, s, 4);
          else if (bits == 8) v = ((int)s[0] - 128) / 128.0f;
          else if (bits == 16) v = (int16_t)(s[0] | s[1] << 8) / 32768.0f;
          else if (bits == 24) v = (float)((int32_t)((uint32_t)s[0] << 8 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 24) >> 8) / 8388608.0f;
          else { int32_t x; memcpy(&x, s, 4); v = x / 2147483648.0f; }
          acc += v;
        }
        mono[i] = acc / ch;
      }
      return true;
    }
    p = body + len + (len & 1);
  }
  throw std::runtime_error("wav: no data chunk");
}

// Any container Media Foundation can read (mp3, m4a/aac, flac, wma, wav, ...), decoded to float, mixed to mono.
// max_seconds > 0: decoding stops past it (the caller refuses the clip; a long file does not fill the memory first)
inline std::vector<float> decode_mf(const std::vector<uint8_t>& bytes, int& rate, double max_seconds = 0) {
  struct Com {
    bool ok;
    bool mf;
    Com() {
      ok = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
      mf = SUCCEEDED(MFStartup(MF_VERSION, MFSTARTUP_LITE));
    }
    ~Com() {
      if (mf) MFShutdown();
      if (ok) CoUninitialize();
    }
  } com;
  auto check = [](HRESULT hr, const char* what) {
    if (FAILED(hr)) throw std::runtime_error(std::string("audio decoding failed (") + what + ")");
  };
  IStream* st = SHCreateMemStream(bytes.data(), (UINT)bytes.size());
  if (!st) throw std::runtime_error("audio: out of memory");
  IMFByteStream* bs = nullptr;
  IMFSourceReader* rd = nullptr;
  IMFMediaType* mt = nullptr;
  IMFMediaType* cur = nullptr;
  std::vector<float> mono;
  try {
    check(MFCreateMFByteStreamOnStream(st, &bs), "byte stream");
    check(MFCreateSourceReaderFromByteStream(bs, nullptr, &rd), "unrecognized audio format");
    check(MFCreateMediaType(&mt), "media type");
    mt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
    mt->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float);
    check(rd->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, mt), "float output");
    check(rd->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, &cur), "output type");
    UINT32 sr = 0, ch = 0;
    cur->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &sr);
    cur->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &ch);
    if (!sr || !ch) throw std::runtime_error("audio: unknown sample rate or channel count");
    rate = (int)sr;
    for (;;) {
      DWORD flags = 0;
      IMFSample* s = nullptr;
      check(rd->ReadSample((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, nullptr, &flags, nullptr, &s), "read");
      if (s) {
        // (the sample, its buffer and the lock let go however the block ends: mono's growth can throw)
        struct Held {
          IMFSample* s;
          IMFMediaBuffer* mb = nullptr;
          bool locked = false;
          ~Held() {
            if (locked) mb->Unlock();
            if (mb) mb->Release();
            s->Release();
          }
        } held{s};
        if (SUCCEEDED(s->ConvertToContiguousBuffer(&held.mb))) {
          BYTE* p = nullptr;
          DWORD len = 0;
          if (SUCCEEDED(held.mb->Lock(&p, nullptr, &len))) {
            held.locked = true;
            const size_t n = len / (4 * ch);
            const float* f = (const float*)p;
            for (size_t i = 0; i < n; ++i) {
              float acc = 0;
              for (UINT32 c = 0; c < ch; ++c) acc += f[i * ch + c];
              mono.push_back(acc / ch);
            }
          }
        }
      }
      if (flags & (MF_SOURCE_READERF_ENDOFSTREAM | MF_SOURCE_READERF_ERROR)) break;
      if (max_seconds > 0 && (double)mono.size() > max_seconds * rate) break;
    }
  } catch (...) {
    if (cur) cur->Release();
    if (mt) mt->Release();
    if (rd) rd->Release();
    if (bs) bs->Release();
    st->Release();
    throw;
  }
  cur->Release();
  mt->Release();
  rd->Release();
  bs->Release();
  st->Release();
  return mono;
}

// Windowed-sinc resampling (low-pass at 0.97 x the lower Nyquist frequency, Blackman window, ~10 lobes per side).
inline std::vector<float> resample(const std::vector<float>& x, int from, int to) {
  if (from == to || x.empty()) return x;
  const double r = (double)to / from, fc = 0.5 * std::min(1.0, r) * 0.97;  // cutoff, cycles per input sample
  const int W = (int)std::ceil(10.0 / (2.0 * fc));                       // half-width in input samples
  const size_t n = (size_t)std::floor((double)x.size() * r);
  std::vector<float> y(n);
  const double pi = 3.14159265358979323846;
  // output i sits at input position i * M / L (the rates over their gcd): its taps depend only on the phase
  // (i * M) % L, so the windowed-sinc weights are computed once per phase (they took a sin and two cos per tap and
  // output sample: 21 s for 14 minutes of 48-kHz MP3 before)
  int64_t g = std::gcd(from, to);
  const int64_t L = to / g, M = from / g, T = 2 * W;
  std::vector<double> tab((size_t)(L * T)), wsum((size_t)L);
  for (int64_t p = 0; p < L; ++p) {
    const double frac = (double)p / L;
    double ws = 0;
    for (int64_t k = 0; k < T; ++k) {
      const double d = frac + W - 1 - k, a = 2 * fc * d;  // (tap k: input sample floor(c) - W + 1 + k)
      const double s = std::fabs(a) < 1e-12 ? 1.0 : std::sin(pi * a) / (pi * a);
      const double t = (d + W) / (2.0 * W);  // window position 0..1
      const double w = (t <= 0 || t >= 1) ? 0 : 0.42 - 0.5 * std::cos(2 * pi * t) + 0.08 * std::cos(4 * pi * t);
      tab[(size_t)(p * T + k)] = 2 * fc * s * w;
      ws += tab[(size_t)(p * T + k)];
    }
    wsum[(size_t)p] = ws;
  }
#pragma omp parallel for if (n > 100000)
  for (int64_t i = 0; i < (int64_t)n; ++i) {
    const int64_t q = i * M, p = q % L, j0 = q / L - W + 1;
    const double* h = &tab[(size_t)(p * T)];
    double acc = 0;
    if (j0 >= 0 && j0 + T <= (int64_t)x.size()) {
      for (int64_t k = 0; k < T; ++k) acc += h[k] * x[(size_t)(j0 + k)];
    } else {
      for (int64_t k = 0; k < T; ++k)
        if (j0 + k >= 0 && j0 + k < (int64_t)x.size()) acc += h[k] * x[(size_t)(j0 + k)];
    }
    y[(size_t)i] = (float)(wsum[(size_t)p] != 0 ? acc / wsum[(size_t)p] : 0);
  }
  return y;
}

// Any supported audio file -> 16 kHz mono samples.
// max_seconds: longer clips are refused (chat audio: 10 minutes, an hours-long clip takes minutes of CPU and more
// context than any model has); 0: no limit (Whisper transcribes recordings of hours)
inline std::vector<float> decode_16k(const std::vector<uint8_t>& bytes, double max_seconds = 600) {
  std::vector<float> s;
  int rate = 0;
  if (!parse_wav(bytes, s, rate)) s = decode_mf(bytes, rate, max_seconds);
  if (s.empty()) throw std::runtime_error("audio: no samples");
  // (a hostile or broken file: an odd sample rate makes the resampling window huge)
  if (rate < 1000 || rate > 384000) throw std::runtime_error("audio: unsupported sample rate " + std::to_string(rate));
  if (max_seconds > 0 && (double)s.size() / rate > max_seconds)
    throw std::runtime_error("audio: longer than " + std::to_string((int)(max_seconds / 60)) + " minutes");
  return resample(s, rate, 16000);
}

}  // namespace audio
