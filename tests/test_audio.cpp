// Checks of src/audio.h without Media Foundation: the streamed resampler against the one-piece resample() (bit for
// bit, any piece sizes and rates), WAV from memory and from a file against the one-piece path, the MP4 audio start
// delay from an edit list, and Ogg Opus repacked as WebM (the packets kept, in order, one SimpleBlock each); since the
// audit of 2026-10-28 also hostile headers (channels, sizes, a wrapping MP4 box), exact sample values and subtitle
// cues of two speakers.
#include "../src/audio.h"
#include "../src/whisper_formats.h"
#include <chrono>
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
  // ---- the audit of 2026-10-28: long inputs, exact sample values, hostile headers
  {
    // the resampler past its buffer trim (~1M input samples: every real file over ~22 s at 48 kHz)
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1, 1);
    bool all = true;
    for (int from : {48000, 44100}) {
      std::vector<float> x(3000001);
      for (float& v : x) v = u(rng);
      const std::vector<float> ref = audio::resample(x, from, 16000);
      audio::Resampler r(from, 16000);
      std::vector<float> y;
      for (size_t a = 0; a < x.size();) {
        const size_t n = std::min<size_t>(x.size() - a, 1 + rng() % 200000);
        r.push(x.data() + a, n, y);
        a += n;
      }
      r.finish(y);
      all = all && y.size() == ref.size() && !memcmp(y.data(), ref.data(), y.size() * 4);
    }
    check("streamed resampling past the buffer trim (3M samples, 48 / 44.1 kHz) = one piece, bit for bit", all);
  }
  {
    // each encoding's sample values, known exactly (not through wav_mono itself)
    struct S { int fmt, bits; std::vector<uint8_t> bytes; float want; };
    const std::vector<S> ss = {
        {1, 8, {192}, 0.5f}, {1, 8, {0}, -1.0f}, {1, 16, {0x00, 0x40}, 0.5f}, {1, 16, {0x00, 0x80}, -1.0f},
        {1, 24, {0x00, 0x00, 0x40}, 0.5f}, {1, 24, {0x00, 0x00, 0xC0}, -0.5f}, {1, 32, {0x00, 0x00, 0x00, 0x40}, 0.5f},
        {3, 32, {0x00, 0x00, 0x80, 0x3E}, 0.25f}};
    bool all = true;
    std::string bad;
    for (const S& s : ss) {
      audio::WavFormat w;
      w.fmt = s.fmt; w.bits = s.bits; w.ch = 1; w.rate = 16000;
      std::vector<float> m;
      audio::wav_mono(w, s.bytes.data(), 1, m);
      if (m.size() != 1 || m[0] != s.want) { all = false; bad = std::to_string(s.bits) + "-bit: " + std::to_string(m.empty() ? 0 : m[0]); }
    }
    audio::WavFormat w2;
    w2.fmt = 1; w2.bits = 16; w2.ch = 2; w2.rate = 16000;
    const uint8_t two[4] = {0x00, 0x40, 0x00, 0xC0};  // (0.5 and -0.5: their mean 0)
    std::vector<float> m2;
    audio::wav_mono(w2, two, 1, m2);
    check("WAV sample values: 8 / 16 / 24 / 32-bit PCM and float, channels averaged", all && m2.size() == 1 && m2[0] == 0.0f, bad);
  }
  {
    // a header of 65,535 channels (gigabytes of buffers): refused; a data size at the field's end (past 4 GB): to the end
    std::vector<float> mono;
    std::vector<uint8_t> b = wav(1, 16, 1, 16000, 1000, mono);
    std::vector<uint8_t> many = b;
    many[22] = 0xFF;
    many[23] = 0xFF;
    bool refused = false;
    try { audio::decode_16k(many, 0); } catch (const std::exception& e) { refused = strstr(e.what(), "channels") != nullptr; }
    check("WAV of 65,535 channels: refused", refused);
    std::vector<uint8_t> big = b;
    big[40] = 0xF0; big[41] = 0xFF; big[42] = 0xFF; big[43] = 0xFF;
    big[4] = 0xFF; big[5] = 0xFF; big[6] = 0xFF; big[7] = 0xFF;
    auto at = [&](uint64_t o, void* d, size_t n) -> size_t {
      if (o >= big.size()) return 0;
      n = (size_t)std::min<uint64_t>(n, big.size() - o);
      memcpy(d, big.data() + o, n);
      return n;
    };
    audio::WavFormat w;
    const bool ok = audio::wav_format(at, big.size(), w);
    check("WAV past 4 GB (sizes at their fields' end): the data to the end of the file", ok && w.len == big.size() - 44,
          std::to_string(w.len));
  }
  {
    // MP4: a 64-bit box size that wraps the sum (an endless loop before); a last box of size 0 (to the end)
    std::vector<uint8_t> f;
    be32(f, 16);
    f.insert(f.end(), {'f', 't', 'y', 'p'});
    for (int i = 0; i < 8; ++i) f.push_back(0);
    be32(f, 1);
    f.insert(f.end(), {'m', 'd', 'a', 't'});
    be32(f, 0xFFFFFFFFu);
    be32(f, 0xFFFFFFF0u);
    for (int i = 0; i < 16; ++i) f.push_back(0);
    auto at = [&](uint64_t o, void* d, size_t n) -> size_t {
      if (o >= f.size()) return 0;
      n = (size_t)std::min<uint64_t>(n, f.size() - o);
      memcpy(d, f.data() + o, n);
      return n;
    };
    const auto t0 = std::chrono::steady_clock::now();
    const double d = audio::mp4_audio_delay(at, f.size());
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    check("MP4: a 64-bit box size that wraps: no delay, at once (it looped forever)", d == 0 && s < 1.0, std::to_string(s) + " s");
  }
  {
    // Ogg: zero lacing values (empty packets) left out; a stream past max_seconds cut there
    std::vector<uint8_t> head = {'O', 'p', 'u', 's', 'H', 'e', 'a', 'd', 1, 1, 0x38, 0x01};
    put32(head, 48000);
    put16(head, 0);
    head.push_back(0);
    const std::vector<uint8_t> tags = {'O', 'p', 'u', 's', 'T', 'a', 'g', 's', 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<uint8_t> ogg;
    page(ogg, 2, 5, 0, lace(head.size()), head);
    page(ogg, 0, 5, 1, lace(tags.size()), tags);
    const std::vector<uint8_t> p20 = {0xF8, 1, 2, 3};  // (CELT 20 ms)
    std::vector<uint8_t> l(200, 0), body;
    l.push_back(4);
    body = p20;
    page(ogg, 0, 5, 2, l, body);  // (200 empty packets, then one)
    for (int k = 0; k < 1000; ++k) page(ogg, 0, 5, 3 + k, lace(4), p20);  // (20 s)
    int opus = 0;
    const std::vector<uint8_t> m = audio::ogg_opus_to_webm(ogg);
    const size_t n_all = blocks(m, 0, m.size(), opus).size();
    const std::vector<uint8_t> m5 = audio::ogg_opus_to_webm(ogg, 5);
    const size_t n5 = blocks(m5, 0, m5.size(), opus).size();
    check("Ogg: empty packets left out (1001 blocks of 1001 real packets)", n_all == 1001, std::to_string(n_all));
    check("Ogg: max_seconds 5 of 20 s: the packets cut at ~6 s", n5 >= 250 && n5 <= 310, std::to_string(n5));
  }
  {
    // cues of two speakers that overlap: in time order, none ending before it starts, a speaker's own never overlapping
    whisper::Result r;
    auto seg = [&](const char* who, double s, double e, std::vector<std::pair<double, const char*>> ws) {
      whisper::Segment g;
      g.start = s;
      g.end = e;
      g.speaker = who;
      for (size_t i = 0; i < ws.size(); ++i) {
        whisper::Word w;
        w.start = ws[i].first;
        w.end = i + 1 < ws.size() ? ws[i + 1].first : e;
        w.text = ws[i].second;
        w.p = 1;
        g.words.push_back(w);
        g.text += std::string(" ") + ws[i].second;
      }
      r.segments.push_back(g);
    };
    seg("Me", 10.0, 19.0, {{10.0, "one"}, {11.0, "two"}, {12.0, "three"}, {13.0, "four."}, {14.4, "five"}, {16.0, "six"},
                           {17.5, "seven"}, {18.5, "eight\n\nnine"}});
    seg("Others", 12.0, 12.5, {{12.0, "Yeah."}});
    whisper::CueOptions co;
    co.max_s = 4.0;
    const auto cues = whisper::make_cues(r, co);
    bool ok = cues.size() >= 3;
    for (size_t i = 0; i < cues.size(); ++i) {
      ok = ok && cues[i].end > cues[i].start && (i == 0 || cues[i].start >= cues[i - 1].start);
      for (size_t j = i + 1; j < cues.size(); ++j)
        if (cues[j].speaker == cues[i].speaker) { ok = ok && cues[i].end <= cues[j].start; break; }
      for (const std::string& line : cues[i].lines) ok = ok && line.find('\n') == std::string::npos;
    }
    const std::string srt = whisper::render("srt", r, co, "t");
    check("cues of two speakers: in order, no negative length, no line break inside a word", ok && srt.find("\n\n\n") == std::string::npos,
          srt.substr(0, 300));
  }
  {
    // Whisper's stock phrases over silence and noise: 20 s of 16-kHz audio, a quiet room (noise at -66 dB) with a
    // "voice" (a 200-Hz tone at -20 dB) from 2 to 6 s and from 12 to 12.8 s (a spoken "terima kasih")
    std::string l;
    check("stock phrases: a credit line (Cyrillic capitals lowered)", whisper::stock_phrase(" Продолжение следует...") == 2 &&
          whisper::stock_phrase("MBC 뉴스 김성현입니다.") == 2 && whisper::stock_phrase("Thanks for watching!") == 2);
    check("stock phrases: thanks with their language", whisper::stock_phrase(" Terima kasih.", &l) == 1 && l == "id" &&
          whisper::stock_phrase("Спасибо.", &l) == 1 && l == "ru" && whisper::stock_phrase(" E aí", &l) == 1 && l == "pt");
    check("stock phrases: speech is not one", !whisper::stock_phrase(" Terima kasih sudah datang ke acara ini") &&
          !whisper::stock_phrase(" Saya pakai bahasa Indonesia"));
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 0.0005f);
    std::vector<float> x(20 * 16000);
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] = nd(rng);
      const double t = (double)i / 16000;
      if ((t >= 2 && t < 6) || (t >= 12 && t < 12.8)) x[i] += 0.14f * (float)std::sin(2 * 3.14159265 * 200 * t);
    }
    auto sg = [](double a, double b, const char* text, const char* lang) {
      whisper::Segment g;
      g.start = a;
      g.end = b;
      g.text = text;
      g.lang = lang;
      return g;
    };
    std::vector<whisper::Segment> s = {sg(2.0, 6.0, " Halo semua, apa kabar", "id"), sg(7.0, 11.0, " Продолжение следует...", "ru"),
                                       sg(12.0, 12.8, " Terima kasih.", "id"), sg(14.0, 16.0, " Gracias.", "es"),
                                       sg(16.0, 19.9, " Saya di sini", "id")};
    whisper::drop_unvoiced(s, x, "");
    std::string got;
    for (const auto& g : s) got += "|" + g.text;
    check("silence / noise: speech and a short thanks in its language kept, the rest left out",
          s.size() == 2 && s[0].text == " Halo semua, apa kabar" && s[1].text == " Terima kasih.", got);
  }
  if (fails) {
    printf("test_audio: %d failed\n", fails);
    return 1;
  }
  printf("test_audio: all passed\n");
  return 0;
}
