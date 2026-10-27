// Audio input: decoding to 16 kHz mono float samples. WAV (PCM 8/16/24/32-bit or float32, any rate and channel
// count) is parsed here; other containers (mp3, m4a/aac, flac, wma, and the audio track of mp4 / mov / mkv / webm /
// wmv / avi) go through Windows Media Foundation; Ogg Opus (voice notes) is repacked as Matroska for it.
// Resampling: windowed-sinc low-pass interpolation (Blackman window), streamed: a file of hours never sits in memory
// at its own rate. Times stay on the file's timeline: an audio track that starts late, or has gaps, gets silence.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <shlwapi.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
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

// ---------------------------------------------------------------- resampling
// the windowed-sinc weights of a rate pair (low-pass at 0.97 x the lower Nyquist frequency, Blackman window, ~10
// lobes per side). Output i sits at input position i * M / L (the rates over their gcd): its taps depend only on the
// phase (i * M) % L, so the weights are computed once per phase (a sin and two cos per tap and output sample took
// 21 s for 14 minutes of 48-kHz MP3)
struct Sinc {
  double r = 1;
  int W = 0;
  int64_t L = 1, M = 1, T = 0;
  std::vector<double> tab, wsum;
  Sinc(int from, int to) {
    r = (double)to / from;
    const double fc = 0.5 * std::min(1.0, r) * 0.97;  // cutoff, cycles per input sample
    W = (int)std::ceil(10.0 / (2.0 * fc));             // half-width in input samples
    const double pi = 3.14159265358979323846;
    const int64_t g = std::gcd(from, to);
    L = to / g;
    M = from / g;
    T = 2 * W;
    tab.resize((size_t)(L * T));
    wsum.resize((size_t)L);
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
  }
  int64_t first_tap(int64_t i) const { return i * M / L - W + 1; }
  // output i from the input samples j < n, held at x[j - base] (the taps past either end left out)
  float out(int64_t i, const float* x, int64_t base, int64_t n) const {
    const int64_t q = i * M, p = q % L, j0 = q / L - W + 1;
    const double* h = &tab[(size_t)(p * T)];
    double acc = 0;
    if (j0 >= 0 && j0 + T <= n) {
      for (int64_t k = 0; k < T; ++k) acc += h[k] * x[j0 + k - base];
    } else {
      for (int64_t k = 0; k < T; ++k)
        if (j0 + k >= 0 && j0 + k < n) acc += h[k] * x[j0 + k - base];
    }
    return (float)(wsum[(size_t)p] != 0 ? acc / wsum[(size_t)p] : 0);
  }
};

// all at once
inline std::vector<float> resample(const std::vector<float>& x, int from, int to) {
  if (from == to || x.empty()) return x;
  const Sinc s(from, to);
  const size_t n = (size_t)std::floor((double)x.size() * s.r);
  std::vector<float> y(n);
#pragma omp parallel for if (n > 100000)
  for (int64_t i = 0; i < (int64_t)n; ++i) y[(size_t)i] = s.out(i, x.data(), 0, (int64_t)x.size());
  return y;
}

// in pieces: the same samples as resample() on the whole (each output computed once its taps are all in, the
// inputs no later output reads let go)
class Resampler {
 public:
  Resampler(int from, int to) {
    if (from != to) s_ = std::make_unique<Sinc>(from, to);
  }
  void push(const float* x, size_t n, std::vector<float>& out) {
    if (!s_) { out.insert(out.end(), x, x + n); return; }
    buf_.insert(buf_.end(), x, x + n);
    total_ += (int64_t)n;
    int64_t last = next_;
    while (s_->first_tap(last) + s_->T <= total_) ++last;
    const size_t o0 = out.size();
    out.resize(o0 + (size_t)(last - next_));
    const float* b = buf_.data();
    const int64_t base = base_, tot = total_, i0 = next_;
#pragma omp parallel for if (last - i0 > 50000)
    for (int64_t i = i0; i < last; ++i) out[o0 + (size_t)(i - i0)] = s_->out(i, b, base, tot);
    next_ = last;
    const int64_t need = std::max<int64_t>(0, s_->first_tap(next_));
    if (need - base_ > (1 << 20)) {
      buf_.erase(buf_.begin(), buf_.begin() + (need - base_));
      base_ = need;
    }
  }
  void finish(std::vector<float>& out) {
    if (!s_) return;
    const int64_t n = (int64_t)std::floor((double)total_ * s_->r);
    for (; next_ < n; ++next_) out.push_back(s_->out(next_, buf_.data(), base_, total_));
  }

 private:
  std::unique_ptr<Sinc> s_;
  std::vector<float> buf_;
  int64_t base_ = 0, total_ = 0, next_ = 0;
};

// decoded pieces (mono, at their rate) -> 16 kHz; a change of rate starts a new resampler. limit > 0: add() says
// false past it (the caller refuses the clip; a long file does not fill the memory first)
class To16k {
 public:
  explicit To16k(double limit_s = 0) : limit_(limit_s) {}
  double seconds = 0;  // decoded so far
  bool add(const float* x, size_t n, int rate) {
    // (a hostile or broken file: an odd sample rate makes the resampling window huge)
    if (rate < 1000 || rate > 384000) throw std::runtime_error("audio: unsupported sample rate " + std::to_string(rate));
    if (rate != rate_) {
      if (rs_) rs_->finish(out_);
      rs_ = std::make_unique<Resampler>(rate, 16000);
      rate_ = rate;
    }
    rs_->push(x, n, out_);
    seconds += (double)n / rate;
    return limit_ <= 0 || seconds <= limit_;
  }
  bool silence(double s, int rate) {  // (in pieces: a long gap does not take its whole length in memory at once)
    if (rate < 1000 || rate > 384000) throw std::runtime_error("audio: unsupported sample rate " + std::to_string(rate));
    const std::vector<float> z((size_t)rate, 0.0f);
    for (int64_t left = std::llround(s * rate); left > 0; left -= rate)
      if (!add(z.data(), (size_t)std::min<int64_t>(left, rate), rate)) return false;
    return true;
  }
  bool over() const { return limit_ > 0 && seconds > limit_; }
  std::vector<float> finish() {
    if (rs_) rs_->finish(out_);
    rs_.reset();
    return std::move(out_);
  }

 private:
  double limit_;
  int rate_ = 0;
  std::unique_ptr<Resampler> rs_;
  std::vector<float> out_;
};

// ---------------------------------------------------------------- WAV
struct WavFormat {
  int fmt = 0, ch = 0, bits = 0, rate = 0;
  uint64_t data = 0, len = 0;  // the data chunk: offset, bytes
};
// the RIFF/WAVE chunks read through `at(offset, dst, n)` (bytes read): false when it is not a WAV file; throws on an
// unsupported WAV encoding
inline bool wav_format(const std::function<size_t(uint64_t, void*, size_t)>& at, uint64_t size, WavFormat& w) {
  uint8_t h[12];
  if (at(0, h, 12) < 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) return false;
  auto u16 = [](const uint8_t* b) { return (uint32_t)b[0] | (uint32_t)b[1] << 8; };
  auto u32 = [&](const uint8_t* b) { return u16(b) | u16(b + 2) << 16; };
  for (uint64_t p = 12; p + 8 <= size;) {
    uint8_t c[8];
    if (at(p, c, 8) < 8) break;
    const uint32_t len = u32(c + 4);
    const uint64_t body = p + 8, avail = std::min<uint64_t>(len, size - body);
    if (!memcmp(c, "fmt ", 4) && avail >= 16) {
      uint8_t f[26] = {};
      at(body, f, (size_t)std::min<uint64_t>(avail, 26));
      w.fmt = (int)u16(f);
      w.ch = (int)u16(f + 2);
      w.rate = (int)u32(f + 4);
      w.bits = (int)u16(f + 14);
      if (w.fmt == 0xFFFE && avail >= 26) w.fmt = (int)u16(f + 24);  // WAVE_FORMAT_EXTENSIBLE: sub-format GUID
    } else if (!memcmp(c, "data", 4)) {
      if (w.ch <= 0 || w.rate <= 0) throw std::runtime_error("wav: data before fmt");
      if (w.ch > 64) throw std::runtime_error("wav: more than 64 channels");
      if (!((w.fmt == 1 && (w.bits == 8 || w.bits == 16 || w.bits == 24 || w.bits == 32)) || (w.fmt == 3 && w.bits == 32)))
        throw std::runtime_error("wav: unsupported encoding (PCM 8/16/24/32-bit or float32 expected)");
      w.data = body;
      // (a size at the 32-bit field's end: a recording past 4 GB, whose header cannot hold its size; its data runs to
      //  the end of the file)
      w.len = len >= 0xFFFFFF00u ? size - body : avail;
      return true;
    }
    p = body + len + (len & 1);
  }
  throw std::runtime_error("wav: no data chunk");
}
// frames of interleaved samples -> mono
inline void wav_mono(const WavFormat& w, const uint8_t* d, size_t frames, std::vector<float>& mono) {
  const int bps = w.bits / 8, ch = w.ch;
  mono.resize(frames);
  for (size_t i = 0; i < frames; ++i) {
    float acc = 0;
    for (int c = 0; c < ch; ++c) {
      const uint8_t* s = d + (i * ch + c) * bps;
      float v;
      if (w.fmt == 3) memcpy(&v, s, 4);
      else if (w.bits == 8) v = ((int)s[0] - 128) / 128.0f;
      else if (w.bits == 16) v = (int16_t)(s[0] | s[1] << 8) / 32768.0f;
      else if (w.bits == 24) v = (float)((int32_t)((uint32_t)s[0] << 8 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 24) >> 8) / 8388608.0f;
      else { int32_t x; memcpy(&x, s, 4); v = x / 2147483648.0f; }
      acc += v;
    }
    mono[i] = acc / ch;
  }
}
// the data chunk through `at`, a block at a time
inline void wav_decode(const std::function<size_t(uint64_t, void*, size_t)>& at, const WavFormat& w, To16k& sink) {
  const size_t frame = (size_t)(w.bits / 8) * w.ch, block = std::max<size_t>(1, (1 << 20) / frame);  // (~1 MB a read)
  std::vector<uint8_t> raw(frame * block);
  std::vector<float> mono;
  for (uint64_t off = 0; off + frame <= w.len;) {
    const size_t want = (size_t)std::min<uint64_t>(raw.size(), (w.len - off) / frame * frame);
    const size_t got = at(w.data + off, raw.data(), want) / frame * frame;
    if (got == 0) break;
    wav_mono(w, raw.data(), got / frame, mono);
    if (!sink.add(mono.data(), mono.size(), w.rate)) return;
    off += got;
  }
}

// ---------------------------------------------------------------- MP4: an audio track that starts late
// The start delay of the first audio track (an empty edit at the head of its edit list): Media Foundation's MP4
// source gives that track's first sample time 0, but players place it after the delay.
inline double mp4_audio_delay(const std::function<size_t(uint64_t, void*, size_t)>& at, uint64_t size) {
  auto be = [](const uint8_t* b, int n) { uint64_t v = 0; for (int i = 0; i < n; ++i) v = v << 8 | b[i]; return v; };
  // the boxes in [a, b): (type, body offset, body size)
  // (a box's size checked against the room left, not a + len: a 64-bit size could wrap the sum and loop forever; at
  //  most 100,000 boxes a level)
  auto boxes = [&](uint64_t a, uint64_t b, const std::function<bool(const char*, uint64_t, uint64_t)>& f) {
    for (int count = 0; a + 8 <= b && count < 100000; ++count) {
      uint8_t h[16];
      if (at(a, h, 8) < 8) return;
      uint64_t len = be(h, 4), hdr = 8;
      if (len == 1) {
        if (at(a + 8, h + 8, 8) < 8) return;
        len = be(h + 8, 8);
        hdr = 16;
      } else if (len == 0) {
        len = b - a;
      }
      if (len < hdr || len > b - a) return;
      char t[5] = {(char)h[4], (char)h[5], (char)h[6], (char)h[7], 0};
      if (!f(t, a + hdr, len - hdr)) return;
      a += len;
    }
  };
  uint64_t moov = 0, moov_len = 0;
  boxes(0, size, [&](const char* t, uint64_t o, uint64_t n) {
    if (strcmp(t, "moov")) return true;
    moov = o;
    moov_len = n;
    return false;
  });
  if (!moov || moov_len > (64u << 20)) return 0;
  double timescale = 0, delay = 0;
  bool found = false;
  boxes(moov, moov + moov_len, [&](const char* t, uint64_t o, uint64_t n) {
    uint8_t b[32] = {};
    if (!strcmp(t, "mvhd") && n >= 32) {
      at(o, b, 32);
      timescale = (double)(b[0] == 1 ? be(b + 20, 4) : be(b + 12, 4));
    } else if (!strcmp(t, "trak")) {
      bool sound = false;
      double d = 0;
      boxes(o, o + n, [&](const char* t2, uint64_t o2, uint64_t n2) {
        if (!strcmp(t2, "mdia"))
          boxes(o2, o2 + n2, [&](const char* t3, uint64_t o3, uint64_t n3) {
            uint8_t hb[12] = {};
            if (!strcmp(t3, "hdlr") && n3 >= 12 && at(o3, hb, 12) == 12 && !memcmp(hb + 8, "soun", 4)) sound = true;
            return true;
          });
        if (!strcmp(t2, "edts"))
          boxes(o2, o2 + n2, [&](const char* t3, uint64_t o3, uint64_t n3) {
            uint8_t e[8];
            if (strcmp(t3, "elst") || n3 < 8 || at(o3, e, 8) < 8) return true;
            const bool v1 = e[0] == 1;
            const uint64_t cnt = be(e + 4, 4), es = v1 ? 20 : 12;
            for (uint64_t k = 0; k < cnt && 8 + (k + 1) * es <= n3 && k < 16; ++k) {  // (empty edits at the head)
              uint8_t x[20];
              if (at(o3 + 8 + k * es, x, (size_t)es) < es) break;
              const int64_t mt = v1 ? (int64_t)be(x + 8, 8) : (int64_t)(int32_t)be(x + 4, 4);
              if (mt != -1) break;
              d += (double)be(x, v1 ? 8 : 4);
            }
            return true;
          });
        return true;
      });
      if (sound) {
        delay = d;
        found = true;
        return false;
      }
    }
    return true;
  });
  return found && timescale > 0 ? delay / timescale : 0;
}

// ---------------------------------------------------------------- Ogg Opus -> Matroska
// Ogg Opus (WhatsApp / Telegram voice notes, .opus / .ogg): Media Foundation has no Ogg reader but decodes Opus in
// Matroska, so the first Opus stream's packets are repacked into a WebM file in memory. Empty: not Ogg.
// max_seconds > 0: the packets past it (and a second) left out, the caller's limit then refuses the clip, so a long or
// hostile file is not repacked whole first. Empty packets are left out (a page of zero lacing values made one each).
inline std::vector<uint8_t> ogg_opus_to_webm(const std::vector<uint8_t>& b, double max_seconds = 0) {
  if (b.size() < 32 || memcmp(b.data(), "OggS", 4)) return {};
  // each packet's length in 48-kHz samples, from its TOC byte (RFC 6716 3.1)
  auto samples = [](const std::vector<uint8_t>& p) -> int64_t {
    if (p.empty()) return 0;
    static const int silk[4] = {480, 960, 1920, 2880}, celt[4] = {120, 240, 480, 960};
    const int cfg = p[0] >> 3, fs = cfg < 12 ? silk[cfg & 3] : cfg < 16 ? ((cfg & 1) ? 960 : 480) : celt[cfg & 3];
    const int c = p[0] & 3, frames = c == 0 ? 1 : c < 3 ? 2 : (p.size() >= 2 ? (p[1] & 0x3F) : 0);
    return (int64_t)fs * frames;
  };
  // (at most 400 packets a second, 2.5-ms frames, past the limit: packets that claim no length count too)
  const int64_t max48 = max_seconds > 0 ? (int64_t)((max_seconds + 1) * 48000) : INT64_MAX,
                max_pk = max_seconds > 0 ? (int64_t)((max_seconds + 1) * 400) + 2 : INT64_MAX;
  int64_t seen48 = 0;
  std::vector<std::vector<uint8_t>> pk;
  std::vector<uint8_t> cur;
  bool have = false, started = false;
  uint32_t serial = 0;
  for (size_t off = 0; off + 27 <= b.size() && seen48 <= max48 && (int64_t)pk.size() <= max_pk;) {
    if (memcmp(b.data() + off, "OggS", 4)) throw std::runtime_error("ogg: broken page");
    const uint32_t sn = b[off + 14] | b[off + 15] << 8 | b[off + 16] << 16 | (uint32_t)b[off + 17] << 24;
    const size_t nseg = b[off + 26], head = off + 27 + nseg;
    if (head > b.size()) break;
    size_t body = 0;
    for (size_t i = 0; i < nseg; ++i) body += b[off + 27 + i];
    if (head + body > b.size()) break;
    if (!have) {  // (the first stream whose first packet is an OpusHead)
      if (body >= 8 && !memcmp(b.data() + head, "OpusHead", 8)) { serial = sn; have = true; }
      else if (body >= 7 && !memcmp(b.data() + head + 1, "vorbis", 6))
        throw std::runtime_error("Ogg Vorbis is not supported (Ogg Opus is)");
    }
    if (have && sn == serial) {
      const bool continued = b[off + 5] & 1;
      auto packet = [&] {
        if (pk.size() >= 2) seen48 += samples(cur);
        pk.push_back(std::move(cur));
        cur.clear();
      };
      if (!continued && !cur.empty()) packet();  // (a packet the stream left unfinished)
      size_t p = head;
      for (size_t i = 0; i < nseg; ++i) {
        const size_t l = b[off + 27 + i];
        if (started || !continued) cur.insert(cur.end(), b.begin() + p, b.begin() + p + l);
        p += l;
        if (l < 255) {
          if (!cur.empty()) packet();
          started = true;
        }
      }
    }
    off = head + body;
  }
  if (!have || pk.size() < 3 || pk[0].size() < 19) throw std::runtime_error("ogg: no Opus stream");
  const std::vector<uint8_t>& oh = pk[0];
  const int channels = oh[9], pre_skip = oh[10] | oh[11] << 8;
  // EBML writing: an element is its ID, its size (8 bytes here), its payload
  auto id = [](std::string& o, uint32_t v) {
    int n = v > 0xFFFFFF ? 4 : v > 0xFFFF ? 3 : v > 0xFF ? 2 : 1;
    for (int i = n - 1; i >= 0; --i) o += (char)(v >> (8 * i) & 0xFF);
  };
  auto el = [&](std::string& o, uint32_t v, const std::string& payload) {
    id(o, v);
    o += (char)0x01;
    for (int i = 6; i >= 0; --i) o += (char)((uint64_t)payload.size() >> (8 * i) & 0xFF);
    o += payload;
  };
  auto un = [&](std::string& o, uint32_t v, uint64_t x) {
    std::string p;
    int n = 1;
    while (n < 8 && (x >> (8 * n))) ++n;
    for (int i = n - 1; i >= 0; --i) p += (char)(x >> (8 * i) & 0xFF);
    el(o, v, p);
  };
  auto fl = [&](std::string& o, uint32_t v, double x) {
    uint64_t u;
    memcpy(&u, &x, 8);
    std::string p;
    for (int i = 7; i >= 0; --i) p += (char)(u >> (8 * i) & 0xFF);
    el(o, v, p);
  };
  std::string ebml, info, track, audio, tracks, clusters, cluster, out;
  un(ebml, 0x4286, 1);
  un(ebml, 0x42F7, 1);
  un(ebml, 0x42F2, 4);
  un(ebml, 0x42F3, 8);
  el(ebml, 0x4282, "webm");
  un(ebml, 0x4287, 4);
  un(ebml, 0x4285, 2);
  int64_t t48 = 0;
  for (size_t i = 2; i < pk.size(); ++i) t48 += samples(pk[i]);
  un(info, 0x2AD7B1, 1000000);  // (timestamps in ms)
  fl(info, 0x4489, (double)t48 / 48.0);
  el(info, 0x4D80, "arcinferzero");
  el(info, 0x5741, "arcinferzero");
  fl(audio, 0xB5, 48000.0);
  un(audio, 0x9F, (uint64_t)channels);
  un(track, 0xD7, 1);
  un(track, 0x73C5, 1);
  un(track, 0x83, 2);
  un(track, 0x9C, 0);
  el(track, 0x86, "A_OPUS");
  el(track, 0x63A2, std::string(oh.begin(), oh.end()));
  un(track, 0x56AA, (uint64_t)pre_skip * 1000000000ull / 48000);
  un(track, 0x56BB, 80000000);
  el(track, 0xE1, audio);
  el(tracks, 0xAE, track);
  int64_t pos = 0, cl_ms = -1;
  auto flush = [&]() {
    if (cl_ms >= 0) el(clusters, 0x1F43B675, cluster);
    cluster.clear();
  };
  for (size_t i = 2; i < pk.size(); ++i) {
    const int64_t ms = (pos + 24) / 48;
    if (cl_ms < 0 || ms - cl_ms > 5000) {
      flush();
      cl_ms = ms;
      un(cluster, 0xE7, (uint64_t)cl_ms);
    }
    const int rel = (int)(ms - cl_ms);
    std::string blk = "\x81";  // (track 1)
    blk += (char)(rel >> 8 & 0xFF);
    blk += (char)(rel & 0xFF);
    blk += '\x80';  // (a keyframe)
    blk.append(pk[i].begin(), pk[i].end());
    el(cluster, 0xA3, blk);
    pos += samples(pk[i]);
  }
  flush();
  pk.clear();
  pk.shrink_to_fit();
  // (the segment written around the clusters, not copied into a string of its own first)
  std::string head;
  el(head, 0x1549A966, info);
  el(head, 0x1654AE6B, tracks);
  el(out, 0x1A45DFA3, ebml);
  id(out, 0x18538067);
  out += (char)0x01;
  const uint64_t seg_len = head.size() + clusters.size();
  for (int i = 6; i >= 0; --i) out += (char)(seg_len >> (8 * i) & 0xFF);
  out += head;
  std::vector<uint8_t> v;
  v.reserve(out.size() + clusters.size());
  v.insert(v.end(), out.begin(), out.end());
  v.insert(v.end(), clusters.begin(), clusters.end());
  return v;
}

// ---------------------------------------------------------------- Media Foundation
struct MfSession {  // (COM and Media Foundation for this thread, while decoding)
  bool ok, mf;
  MfSession() {
    ok = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
    mf = SUCCEEDED(MFStartup(MF_VERSION, MFSTARTUP_LITE));
  }
  ~MfSession() {
    if (mf) MFShutdown();
    if (ok) CoUninitialize();
  }
};
template <class T> struct ComPtr {
  T* p = nullptr;
  ~ComPtr() { if (p) p->Release(); }
  T** operator&() { return &p; }
  T* operator->() const { return p; }
};
inline void mf_check(HRESULT hr, const char* what) {
  if (FAILED(hr)) throw std::runtime_error(std::string("audio decoding failed (") + what + ")");
}

// the first audio stream of a source reader, as float, mixed to mono, into the sink. Each sample is placed at its
// time (+ delay): a late start or a gap of more than 0.1 s becomes silence, so the times stay on the file's
// timeline (one larger than 10 minutes is taken for a jump of the clock, not a gap; smaller ones are codec frames:
// Media Foundation's MP3 decoder drops the encoder delay from its first buffer, ~12 ms, and keeps it in the next
// one's time; MP3 in AVI starts one 36-ms frame in). The format is read again when the decoder changes it (an AAC
// stream's real rate and channels come with its first sample).
inline void mf_decode(IMFSourceReader* rd, To16k& sink, double delay) {
  // (the audio stream alone: a selected video stream that is never read keeps its samples queued, gigabytes over an
  //  hours-long video)
  rd->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, FALSE);
  mf_check(rd->SetStreamSelection((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, TRUE), "no audio stream");
  ComPtr<IMFMediaType> mt;
  mf_check(MFCreateMediaType(&mt), "media type");
  mt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
  mt->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float);
  mf_check(rd->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, mt.p), "float output");
  UINT32 sr = 0, ch = 0;
  auto format = [&]() {
    ComPtr<IMFMediaType> cur;
    mf_check(rd->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, &cur), "output type");
    sr = ch = 0;
    cur->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &sr);
    cur->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &ch);
    if (!sr || !ch) throw std::runtime_error("audio: unknown sample rate or channel count");
    if (sr < 1000 || sr > 384000 || ch > 64) throw std::runtime_error("audio: unsupported format (" + std::to_string(sr) +
                                                                     " Hz, " + std::to_string(ch) + " channels)");
  };
  format();
  double pos = 0;  // (seconds placed so far)
  std::vector<float> mono;
  for (;;) {
    DWORD flags = 0;
    LONGLONG ts = 0;
    ComPtr<IMFSample> s;
    mf_check(rd->ReadSample((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, nullptr, &flags, &ts, &s), "read");
    if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) format();
    if (s.p) {
      ComPtr<IMFMediaBuffer> mb;
      if (SUCCEEDED(s->ConvertToContiguousBuffer(&mb))) {
        BYTE* p = nullptr;
        DWORD len = 0;
        if (SUCCEEDED(mb->Lock(&p, nullptr, &len))) {
          const size_t n = len / (4 * ch);
          const float* f = (const float*)p;
          mono.resize(n);
          for (size_t i = 0; i < n; ++i) {
            float acc = 0;
            for (UINT32 c = 0; c < ch; ++c) acc += f[i * ch + c];
            mono[i] = acc / ch;
          }
          mb->Unlock();
          const double t = delay + ts / 1e7;
          bool more = true;
          if (t - pos > 0.1 && t - pos < 600) more = sink.silence(t - pos, (int)sr);
          if (t - pos > 0.1) pos = t;
          if (more) more = sink.add(mono.data(), n, (int)sr);
          pos += (double)n / sr;
          if (!more) break;
        }
      }
    }
    if (flags & (MF_SOURCE_READERF_ENDOFSTREAM | MF_SOURCE_READERF_ERROR)) break;
  }
}

inline void mf_decode_bytes(const std::vector<uint8_t>& bytes, To16k& sink, double delay) {
  MfSession ses;
  ComPtr<IStream> st;
  st.p = SHCreateMemStream(bytes.data(), (UINT)bytes.size());
  if (!st.p) throw std::runtime_error("audio: out of memory");
  ComPtr<IMFByteStream> bs;
  ComPtr<IMFSourceReader> rd;
  mf_check(MFCreateMFByteStreamOnStream(st.p, &bs), "byte stream");
  mf_check(MFCreateSourceReaderFromByteStream(bs.p, nullptr, &rd), "unrecognized audio format");
  mf_decode(rd.p, sink, delay);
}

// ---------------------------------------------------------------- 16 kHz mono
// Any supported audio file -> 16 kHz mono samples.
// max_seconds: longer clips are refused (chat audio: 10 minutes, an hours-long clip takes minutes of CPU and more
// context than any model has); 0: no limit (Whisper transcribes recordings of hours)
inline std::vector<float> finish_16k(To16k& sink, double max_seconds) {
  if (sink.over()) throw std::runtime_error("audio: longer than " + std::to_string((int)(max_seconds / 60)) + " minutes");
  std::vector<float> s = sink.finish();
  if (s.empty()) throw std::runtime_error("audio: no samples");
  return s;
}

// from memory (an upload)
inline std::vector<float> decode_16k(const std::vector<uint8_t>& bytes, double max_seconds = 600) {
  To16k sink(max_seconds);
  auto at = [&](uint64_t o, void* d, size_t n) -> size_t {
    if (o >= bytes.size()) return 0;
    n = (size_t)std::min<uint64_t>(n, bytes.size() - o);
    memcpy(d, bytes.data() + o, n);
    return n;
  };
  WavFormat w;
  if (wav_format(at, bytes.size(), w)) {
    wav_decode(at, w, sink);
  } else {
    const std::vector<uint8_t> webm = ogg_opus_to_webm(bytes, max_seconds);
    if (!webm.empty()) mf_decode_bytes(webm, sink, 0);
    else mf_decode_bytes(bytes, sink, mp4_audio_delay(at, bytes.size()));
  }
  return finish_16k(sink, max_seconds);
}

// from a file, read as it goes (hours of video or a WAV of gigabytes never sit in memory whole)
inline std::vector<float> decode_16k_file(const std::filesystem::path& path, double max_seconds = 0) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path.string());
  const uint64_t size = std::filesystem::file_size(path);
  auto at = [&](uint64_t o, void* d, size_t n) -> size_t {
    f.clear();
    f.seekg((std::streamoff)o);
    f.read((char*)d, (std::streamsize)n);
    return (size_t)f.gcount();
  };
  To16k sink(max_seconds);
  WavFormat w;
  uint8_t magic[4] = {};
  at(0, magic, 4);
  if (wav_format(at, size, w)) {
    wav_decode(at, w, sink);
  } else if (!memcmp(magic, "OggS", 4)) {
    if (size > (1u << 30)) throw std::runtime_error("ogg: larger than 1 GB");
    std::vector<uint8_t> b((size_t)size);
    at(0, b.data(), b.size());
    mf_decode_bytes(ogg_opus_to_webm(b, max_seconds), sink, 0);
  } else {
    const double delay = mp4_audio_delay(at, size);
    MfSession ses;
    ComPtr<IMFSourceReader> rd;
    mf_check(MFCreateSourceReaderFromURL(std::filesystem::absolute(path).wstring().c_str(), nullptr, &rd),
             "unrecognized audio format");
    mf_decode(rd.p, sink, delay);
  }
  return finish_16k(sink, max_seconds);
}

// an audio or video file's length in seconds from its container (no decoding; 0: not known)
inline double duration_seconds(const std::filesystem::path& path) {
  try {
    MfSession ses;
    ComPtr<IMFSourceReader> rd;
    if (FAILED(MFCreateSourceReaderFromURL(std::filesystem::absolute(path).wstring().c_str(), nullptr, &rd))) return 0;
    PROPVARIANT v;
    PropVariantInit(&v);
    double s = 0;
    if (SUCCEEDED(rd->GetPresentationAttribute((DWORD)MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &v)) && v.vt == VT_UI8)
      s = (double)v.uhVal.QuadPart / 1e7;
    PropVariantClear(&v);
    return s;
  } catch (const std::exception&) {
    return 0;
  }
}

// a WAV file's channels apart, each at 16 kHz (a recording's tracks: the microphone, the other side); max_seconds > 0:
// longer refused, as decode_16k_file
inline std::vector<std::vector<float>> decode_16k_wav_channels(const std::filesystem::path& path, double max_seconds = 0) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path.string());
  const uint64_t size = std::filesystem::file_size(path);
  auto at = [&](uint64_t o, void* d, size_t n) -> size_t {
    f.clear();
    f.seekg((std::streamoff)o);
    f.read((char*)d, (std::streamsize)n);
    return (size_t)f.gcount();
  };
  WavFormat w;
  if (!wav_format(at, size, w)) throw std::runtime_error("not a WAV file");
  WavFormat one = w;  // (each channel read as a mono file of its own)
  one.ch = 1;
  std::vector<To16k> sinks;
  for (int c = 0; c < w.ch; ++c) sinks.emplace_back(max_seconds);
  const size_t bps = (size_t)w.bits / 8, frame = bps * w.ch, block = std::max<size_t>(1, (1 << 20) / frame);
  std::vector<uint8_t> raw(frame * block), ch(bps * block);
  std::vector<float> mono;
  for (uint64_t off = 0; off + frame <= w.len;) {
    const size_t want = (size_t)std::min<uint64_t>(raw.size(), (w.len - off) / frame * frame);
    const size_t got = at(w.data + off, raw.data(), want) / frame * frame, n = got / frame;
    if (!n) break;
    bool more = true;
    for (int c = 0; c < w.ch; ++c) {
      for (size_t i = 0; i < n; ++i) memcpy(&ch[i * bps], &raw[i * frame + c * bps], bps);
      wav_mono(one, ch.data(), n, mono);
      more = sinks[c].add(mono.data(), n, w.rate) && more;
    }
    if (!more) break;
    off += got;
  }
  std::vector<std::vector<float>> out;
  for (To16k& s : sinks) out.push_back(finish_16k(s, max_seconds));
  return out;
}

// a 16-bit PCM WAV as MP3 (Windows' own MP3 encoder through Media Foundation; 32 / 44.1 / 48 kHz): kbps 96-320.
// False and the reason when it cannot (the WAV is then kept)
inline bool encode_mp3(const std::filesystem::path& wav, const std::filesystem::path& mp3, int kbps, std::string& err) {
  try {
    std::ifstream f(wav, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + wav.string());
    const uint64_t size = std::filesystem::file_size(wav);
    auto at = [&](uint64_t o, void* d, size_t n) -> size_t {
      f.clear();
      f.seekg((std::streamoff)o);
      f.read((char*)d, (std::streamsize)n);
      return (size_t)f.gcount();
    };
    WavFormat w;
    if (!wav_format(at, size, w) || w.fmt != 1 || w.bits != 16 || w.ch < 1 || w.ch > 2)
      throw std::runtime_error("only 16-bit PCM WAV, mono or stereo");
    MfSession ses;
    // (the encoder takes only some bitrates for a channel count: mono at 192 kbps is refused; the next lower one tried)
    ComPtr<IMFSinkWriter> sw;
    DWORD stream = 0;
    const int top = std::clamp(kbps, 96, 320);
    bool set = false;
    for (int b : {top, 160, 128, 112, 96}) {
      if (b > top) continue;
      if (sw.p) { sw.p->Release(); sw.p = nullptr; }
      ComPtr<IMFMediaType> out, in;
      mf_check(MFCreateSinkWriterFromURL(std::filesystem::absolute(mp3).wstring().c_str(), nullptr, nullptr, &sw), "MP3 file");
      mf_check(MFCreateMediaType(&out), "media type");
      out->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
      out->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_MP3);
      out->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, (UINT32)w.rate);
      out->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, (UINT32)w.ch);
      out->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, (UINT32)b * 1000 / 8);
      if (FAILED(sw->AddStream(out.p, &stream))) continue;
      mf_check(MFCreateMediaType(&in), "media type");
      in->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
      in->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM);
      in->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, (UINT32)w.rate);
      in->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, (UINT32)w.ch);
      in->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
      in->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, (UINT32)(2 * w.ch));
      in->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, (UINT32)(2 * w.ch * w.rate));
      in->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE);
      if (SUCCEEDED(sw->SetInputMediaType(stream, in.p, nullptr))) { set = true; break; }
    }
    if (!set) throw std::runtime_error("the MP3 encoder takes no bitrate for " + std::to_string(w.ch) + " channel(s) at " +
                                       std::to_string(w.rate) + " Hz (32 / 44.1 / 48 kHz)");
    mf_check(sw->BeginWriting(), "begin");
    const size_t frame = (size_t)2 * w.ch, block = (size_t)w.rate / 2;  // (half a second a sample)
    std::vector<uint8_t> raw(frame * block);
    LONGLONG t = 0;
    for (uint64_t off = 0; off + frame <= w.len;) {
      const size_t want = (size_t)std::min<uint64_t>(raw.size(), (w.len - off) / frame * frame);
      const size_t got = at(w.data + off, raw.data(), want) / frame * frame;
      if (!got) break;
      ComPtr<IMFMediaBuffer> mb;
      mf_check(MFCreateMemoryBuffer((DWORD)got, &mb), "buffer");
      BYTE* p = nullptr;
      mb->Lock(&p, nullptr, nullptr);
      memcpy(p, raw.data(), got);
      mb->Unlock();
      mb->SetCurrentLength((DWORD)got);
      ComPtr<IMFSample> s;
      mf_check(MFCreateSample(&s), "sample");
      s->AddBuffer(mb.p);
      const LONGLONG d = (LONGLONG)(got / frame) * 10000000LL / w.rate;
      s->SetSampleTime(t);
      s->SetSampleDuration(d);
      mf_check(sw->WriteSample(stream, s.p), "write");
      t += d;
      off += got;
    }
    mf_check(sw->Finalize(), "finalize");
    return true;
  } catch (const std::exception& e) {
    err = e.what();
    std::error_code ec;
    std::filesystem::remove(mp3, ec);
    return false;
  }
}

}  // namespace audio
