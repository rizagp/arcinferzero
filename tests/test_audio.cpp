// Checks of src/audio.h without Media Foundation: the streamed resampler against the one-piece resample() (bit for
// bit, any piece sizes and rates), WAV from memory and from a file against the one-piece path, the MP4 audio start
// delay from an edit list, and Ogg Opus repacked as WebM (the packets kept, in order, one SimpleBlock each).
#include "../src/audio.h"
#include <cstdio>
#include <random>

static int fails = 0;
static void check(const char* name, bool ok, const std::string& got = "") {
  printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name, ok || got.empty() ? "" : "  got: ", ok ? "" : got.c_str());
  fails += !ok;
}

static void put16(std::vector<uint8_t>& b, uint32_t v) { b.push_back(v & 0xFF); b.push_back(v >> 8 & 0xFF); }
static void put32(std::vector<uint8_t>& b, uint32_t v) { put16(b, v & 0xFFFF); put16(b, v >> 16); }
static void be32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 3; i >= 0; --i) b.push_back(v >> (8 * i) & 0xFF); }

// a WAV of `frames` frames: fmt 1 (PCM) or 3 (float), bits, channels, rate; samples from a seeded sine + noise
static std::vector<uint8_t> wav(int fmt, int bits, int ch, int rate, size_t frames, std::vector<float>& mono) {
  std::vector<uint8_t> d;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-0.2f, 0.2f);
  std::vector<uint8_t> data;
  for (size_t i = 0; i < frames; ++i)
    for (int c = 0; c < ch; ++c) {
      const float v = 0.6f * std::sin(0.01f * (float)i * (c + 1)) + u(rng);
      if (fmt == 3) { uint32_t x; memcpy(&x, &v, 4); put32(data, x); }
      else if (bits == 16) put16(data, (uint16_t)(int16_t)std::lround(v * 32767));
      else if (bits == 24) { const int32_t x = (int32_t)std::lround(v * 8388607); data.push_back(x & 0xFF); data.push_back(x >> 8 & 0xFF); data.push_back(x >> 16 & 0xFF); }
      else if (bits == 8) data.push_back((uint8_t)(128 + std::lround(v * 127)));
    }
  d.insert(d.end(), {'R', 'I', 'F', 'F'});
  put32(d, (uint32_t)(36 + data.size()));
  d.insert(d.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
  put32(d, 16);
  put16(d, (uint32_t)fmt);
  put16(d, (uint32_t)ch);
  put32(d, (uint32_t)rate);
  put32(d, (uint32_t)(rate * ch * bits / 8));
  put16(d, (uint32_t)(ch * bits / 8));
  put16(d, (uint32_t)bits);
  d.insert(d.end(), {'d', 'a', 't', 'a'});
  put32(d, (uint32_t)data.size());
  d.insert(d.end(), data.begin(), data.end());
  audio::WavFormat w;
  w.fmt = fmt; w.bits = bits; w.ch = ch; w.rate = rate;
  audio::wav_mono(w, data.data(), frames, mono);
  return d;
}

// an Ogg page of one stream: its segments as given (lacing values), the packets' bytes after them
static void page(std::vector<uint8_t>& o, uint8_t type, uint32_t serial, uint32_t seq, const std::vector<uint8_t>& lacing,
                 const std::vector<uint8_t>& body) {
  o.insert(o.end(), {'O', 'g', 'g', 'S', 0, type});
  for (int i = 0; i < 8; ++i) o.push_back(0);
  put32(o, serial);
  put32(o, seq);
  put32(o, 0);  // (the CRC: not checked)
  o.push_back((uint8_t)lacing.size());
  o.insert(o.end(), lacing.begin(), lacing.end());
  o.insert(o.end(), body.begin(), body.end());
}
static std::vector<uint8_t> lace(size_t n) {  // (a whole packet of n bytes)
  std::vector<uint8_t> l(n / 255, 255);
  l.push_back((uint8_t)(n % 255));
  return l;
}

// the SimpleBlocks' payloads of our WebM (8-byte element sizes)
static std::vector<std::vector<uint8_t>> blocks(const std::vector<uint8_t>& m, size_t a, size_t b, int& found_opus) {
  std::vector<std::vector<uint8_t>> out;
  while (a < b) {
    uint32_t id = m[a];
    int il = m[a] >= 0x80 ? 1 : m[a] >= 0x40 ? 2 : m[a] >= 0x20 ? 3 : 4;
    for (int i = 1; i < il; ++i) id = id << 8 | m[a + i];
    a += il;
    if (m[a] != 0x01) return out;  // (always 8-byte sizes here)
    uint64_t n = 0;
    for (int i = 1; i < 8; ++i) n = n << 8 | m[a + i];
    a += 8;
    if (id == 0x18538067 || id == 0x1F43B675 || id == 0x1654AE6B || id == 0xAE) {
      auto in = blocks(m, a, a + n, found_opus);
      out.insert(out.end(), in.begin(), in.end());
    } else if (id == 0xA3) {
      out.emplace_back(m.begin() + a + 4, m.begin() + a + n);
    } else if (id == 0x86 && n == 6 && !memcmp(&m[a], "A_OPUS", 6)) {
      ++found_opus;
    }
    a += n;
  }
  return out;
}

int main() {
  // ---- the streamed resampler = resample() on the whole
  {
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-1, 1);
    bool all = true;
    std::string bad;
    for (int from : {48000, 44100, 22050, 32000, 8000, 11025, 16000, 96000}) {
      for (size_t len : {(size_t)5, (size_t)70, (size_t)4000, (size_t)123457}) {
        std::vector<float> x(len);
        for (float& v : x) v = u(rng);
        const std::vector<float> ref = audio::resample(x, from, 16000);
        for (int trial = 0; trial < 3; ++trial) {
          audio::Resampler r(from, 16000);
          std::vector<float> y;
          for (size_t a = 0; a < len;) {
            const size_t n = std::min<size_t>(len - a, 1 + rng() % (trial == 0 ? 7 : trial == 1 ? 900 : 70000));
            r.push(x.data() + a, n, y);
            a += n;
          }
          r.finish(y);
          if (y.size() != ref.size() || memcmp(y.data(), ref.data(), y.size() * 4)) {
            all = false;
            bad = std::to_string(from) + " Hz, " + std::to_string(len) + " samples: " + std::to_string(y.size()) +
                  " vs " + std::to_string(ref.size());
          }
        }
      }
    }
    check("streamed resampling = one piece, bit for bit (8 rates, 4 lengths, 3 piece sizes)", all, bad);
  }
  // ---- WAV, memory and file, = the one-piece path
  {
    struct F { int fmt, bits, ch, rate; };
    bool all = true;
    std::string bad;
    const auto tmp = std::filesystem::temp_directory_path() / "arcinferzero_test_audio.wav";
    for (const F& f : {F{1, 16, 2, 48000}, F{1, 24, 1, 44100}, F{3, 32, 2, 22050}, F{1, 8, 1, 16000}, F{1, 16, 6, 96000}}) {
      std::vector<float> mono;
      const std::vector<uint8_t> b = wav(f.fmt, f.bits, f.ch, f.rate, 200003, mono);
      const std::vector<float> ref = audio::resample(mono, f.rate, 16000);
      const std::vector<float> m = audio::decode_16k(b, 0);
      std::ofstream(tmp, std::ios::binary).write((const char*)b.data(), (std::streamsize)b.size());
      const std::vector<float> fl = audio::decode_16k_file(tmp, 0);
      if (m != ref || fl != ref) {
        all = false;
        bad = std::to_string(f.bits) + "-bit " + std::to_string(f.rate) + " Hz";
      }
    }
    std::filesystem::remove(tmp);
    check("WAV from memory and from a file = one-piece resampling (5 encodings)", all, bad);
    std::vector<float> mono;
    const std::vector<uint8_t> b = wav(1, 16, 1, 48000, 48000 * 3, mono);
    bool refused = false;
    try { audio::decode_16k(b, 2.0); } catch (const std::exception& e) { refused = strstr(e.what(), "longer than") != nullptr; }
    check("a clip past max_seconds is refused", refused);
  }
  // ---- MP4: the first audio track's empty edit
  {
    auto box = [](const char* t, const std::vector<uint8_t>& body) {
      std::vector<uint8_t> o;
      be32(o, (uint32_t)(8 + body.size()));
      o.insert(o.end(), t, t + 4);
      o.insert(o.end(), body.begin(), body.end());
      return o;
    };
    auto cat = [](std::initializer_list<std::vector<uint8_t>> l) {
      std::vector<uint8_t> o;
      for (const auto& x : l) o.insert(o.end(), x.begin(), x.end());
      return o;
    };
    std::vector<uint8_t> mvhd(100, 0);
    mvhd[15] = 0;  // (version 0: timescale at 12)
    mvhd[12] = 0; mvhd[13] = 0; mvhd[14] = 0x03; mvhd[15] = 0xE8;  // 1000
    auto hdlr = [&](const char* h) { std::vector<uint8_t> b(24, 0); memcpy(&b[8], h, 4); return box("hdlr", b); };
    auto elst = [&](std::vector<std::pair<uint32_t, int32_t>> e) {
      std::vector<uint8_t> b(4, 0);
      be32(b, (uint32_t)e.size());
      for (auto& [d, t] : e) { be32(b, d); be32(b, (uint32_t)t); be32(b, 0x00010000); }
      return box("edts", box("elst", b));
    };
    const std::vector<uint8_t> video = box("trak", cat({elst({{500, -1}, {9000, 0}}), box("mdia", hdlr("vide"))}));
    const std::vector<uint8_t> sound = box("trak", cat({box("mdia", hdlr("soun")), elst({{1500, -1}, {500, -1}, {9000, 1024}})}));
    const std::vector<uint8_t> file = cat({box("ftyp", std::vector<uint8_t>(8, 0)), box("mdat", std::vector<uint8_t>(64, 0)),
                                           box("moov", cat({box("mvhd", mvhd), video, sound}))});
    auto at = [&](uint64_t o, void* d, size_t n) -> size_t {
      if (o >= file.size()) return 0;
      n = (size_t)std::min<uint64_t>(n, file.size() - o);
      memcpy(d, file.data() + o, n);
      return n;
    };
    const double d = audio::mp4_audio_delay(at, file.size());
    check("MP4: the sound track's empty edits = its start delay (2.0 s; the video track's 0.5 s ignored)", std::fabs(d - 2.0) < 1e-9,
          std::to_string(d));
    const std::vector<uint8_t> junk(300, 0x41);
    auto at2 = [&](uint64_t o, void* dd, size_t n) -> size_t {
      if (o >= junk.size()) return 0;
      n = (size_t)std::min<uint64_t>(n, junk.size() - o);
      memcpy(dd, junk.data() + o, n);
      return n;
    };
    check("MP4: not an MP4 = no delay", audio::mp4_audio_delay(at2, junk.size()) == 0);
  }
  // ---- Ogg Opus -> WebM
  {
    std::vector<uint8_t> head = {'O', 'p', 'u', 's', 'H', 'e', 'a', 'd', 1, 2, 0x38, 0x01};
    put32(head, 48000);
    put16(head, 0);
    head.push_back(0);
    const std::vector<uint8_t> tags = {'O', 'p', 'u', 's', 'T', 'a', 'g', 's', 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<std::vector<uint8_t>> pk;
    for (size_t n : {(size_t)40, (size_t)300, (size_t)520, (size_t)7}) {
      std::vector<uint8_t> p(n);
      p[0] = 0xF8;  // (CELT 20 ms, one frame)
      for (size_t i = 1; i < n; ++i) p[i] = (uint8_t)(i * 7 + n);
      pk.push_back(p);
    }
    std::vector<uint8_t> ogg;
    page(ogg, 2, 77, 0, lace(head.size()), head);
    page(ogg, 0, 99, 0, lace(5), {1, 2, 3, 4, 5});  // (another stream between: left out)
    page(ogg, 0, 77, 1, lace(tags.size()), tags);
    {  // packets 0 and 1 whole, packet 2 split over two pages
      std::vector<uint8_t> l = lace(pk[0].size()), body = pk[0];
      for (uint8_t x : lace(pk[1].size())) l.push_back(x);
      body.insert(body.end(), pk[1].begin(), pk[1].end());
      l.push_back(255);
      body.insert(body.end(), pk[2].begin(), pk[2].begin() + 255);
      page(ogg, 0, 77, 2, l, body);
      std::vector<uint8_t> rest(pk[2].begin() + 255, pk[2].end());
      std::vector<uint8_t> l2 = lace(rest.size());
      for (uint8_t x : lace(pk[3].size())) l2.push_back(x);
      rest.insert(rest.end(), pk[3].begin(), pk[3].end());
      page(ogg, 1 | 4, 77, 3, l2, rest);
    }
    const std::vector<uint8_t> m = audio::ogg_opus_to_webm(ogg);
    int opus = 0;
    const auto got = m.size() > 4 && !memcmp(m.data(), "\x1A\x45\xDF\xA3", 4) ? blocks(m, 0, m.size(), opus)
                                                                                : std::vector<std::vector<uint8_t>>{};
    check("Ogg Opus -> WebM: an A_OPUS track, the 4 packets in order (one across pages), the other stream left out",
          opus == 1 && got == pk, std::to_string(got.size()) + " blocks, " + std::to_string(opus) + " Opus tracks");
    check("not Ogg -> empty", audio::ogg_opus_to_webm(std::vector<uint8_t>(64, 0)).empty());
    std::vector<uint8_t> vorbis;
    const std::vector<uint8_t> vh = {1, 'v', 'o', 'r', 'b', 'i', 's', 0, 0, 0, 0};
    page(vorbis, 2, 5, 0, lace(vh.size()), vh);
    bool said = false;
    try { audio::ogg_opus_to_webm(vorbis); } catch (const std::exception& e) { said = strstr(e.what(), "Vorbis") != nullptr; }
    check("Ogg Vorbis: refused with its name", said);
  }
  if (fails) {
    printf("test_audio: %d failed\n", fails);
    return 1;
  }
  printf("test_audio: all passed\n");
  return 0;
}
