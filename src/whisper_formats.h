// Transcript formats (src/whisper.h's Result): the full JSON (the source every other format is made from), OpenAI's
// json / verbose_json / text, the faster-whisper-style JSON the Remotion channel pipeline reads ("fw"), Remotion's
// Caption[] ("captions"), and subtitles cut from the word timestamps: SRT, WebVTT (with word timing), YouTube SBV and
// SRV3 (word-by-word), ASS (karaoke \k), TTML; plain text, Markdown, CSV / TSV.
#pragma once
#include "whisper.h"
#include <cstdio>
#include <string>
#include <vector>

namespace whisper {

struct CueOptions {
  int max_chars = 42, max_lines = 2;
  double max_s = 7.0, min_s = 1.0, pause_s = 0.5;
  bool one_word = false;  // a cue per word (vertical short-form video)
};

inline std::string json_str(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
    else if (c == '\n') o += "\\n";
    else if (c == '\r') o += "\\r";
    else if (c == '\t') o += "\\t";
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += (char)c;
  }
  return o + "\"";
}
inline std::string num(double v, int prec = 3) {
  char b[64];
  snprintf(b, sizeof b, "%.*f", prec, v);
  return b;
}
inline std::string xml(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '&') o += "&amp;";
    else if (c == '<') o += "&lt;";
    else if (c == '>') o += "&gt;";
    else if (c == '"') o += "&quot;";
    else o += c;
  }
  return o;
}
inline std::string trim_copy(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && isspace((unsigned char)s[a])) ++a;
  while (b > a && isspace((unsigned char)s[b - 1])) --b;
  return s.substr(a, b - a);
}
// times: h:mm:ss with a separator and digits of fractions (SRT ",", 3; VTT ".", 3; ASS ".", 2)
inline std::string clock(double t, char sep, int frac_digits, bool two_digit_hours = true) {
  if (t < 0) t = 0;
  const long long ms = (long long)std::llround(t * 1000);
  const long long h = ms / 3600000, m = ms / 60000 % 60, s = ms / 1000 % 60, f = ms % 1000;
  char b[48];
  if (frac_digits == 2) snprintf(b, sizeof b, two_digit_hours ? "%02lld:%02lld:%02lld%c%02lld" : "%lld:%02lld:%02lld%c%02lld", h, m, s, sep, f / 10);
  else snprintf(b, sizeof b, two_digit_hours ? "%02lld:%02lld:%02lld%c%03lld" : "%lld:%02lld:%02lld%c%03lld", h, m, s, sep, f);
  return b;
}

// ---------------------------------------------------------------- subtitle cues from the words
struct Cue {
  double start = 0, end = 0;
  std::vector<Word> words;
  std::vector<std::string> lines;
  std::vector<size_t> first;  // (each line's first word)
  std::string speaker;        // (its segments' track, when the recording has several)
};
// words [a, b) as lines of at most max_chars (a longer word alone on its line): the lines' first words, or empty
// when they take more than max_lines. Two lines: the split that evens them out (the top one not longer than needed)
inline std::vector<size_t> line_breaks(const std::vector<Word>& w, size_t a, size_t b, int max_chars, int max_lines) {
  auto len = [&](size_t i, size_t j) {  // (words i..j-1 joined by spaces)
    size_t n = 0;
    for (size_t k = i; k < j; ++k) n += w[k].text.size() + (k > i ? 1 : 0);
    return n;
  };
  if (a >= b) return {a};
  if (len(a, b) <= (size_t)max_chars) return {a};
  if (max_lines == 2) {
    size_t best = 0, score = SIZE_MAX;
    for (size_t k = a + 1; k < b; ++k) {
      const size_t l1 = len(a, k), l2 = len(k, b);
      if ((l1 > (size_t)max_chars && k > a + 1) || (l2 > (size_t)max_chars && b > k + 1)) continue;
      const size_t s = std::max(l1, l2) * 2 + (l1 > l2 ? 1 : 0);  // (even; a tie: the longer line below)
      if (s < score) { score = s; best = k; }
    }
    if (best) return {a, best};
    return {};
  }
  std::vector<size_t> f = {a};
  size_t n = 0;
  for (size_t k = a; k < b; ++k) {
    const size_t add = w[k].text.size() + (k > f.back() ? 1 : 0);
    if (k > f.back() && n + add > (size_t)max_chars) { f.push_back(k); n = w[k].text.size(); }
    else n += add;
  }
  return (int)f.size() <= max_lines ? f : std::vector<size_t>{};
}
inline std::vector<Cue> make_cues(const Result& r, const CueOptions& o) {
  std::vector<Word> all;
  std::vector<const std::string*> who;  // (each word's speaker: a cue holds one)
  for (const Segment& g : r.segments) {
    if (!g.words.empty()) {
      // (a word's own line breaks or tabs as one space: a blank line inside a word ended an SRT / VTT cue there, and
      //  broke an ASS line; a word of white space only left out)
      for (const Word& w : g.words) {
        Word x = w;
        x.text.clear();
        bool sp = false;
        for (char c : w.text) {
          if (isspace((unsigned char)c)) { sp = !x.text.empty(); continue; }
          if (sp) x.text += ' ';
          sp = false;
          x.text += c;
        }
        if (!x.text.empty()) all.push_back(x);
      }
    } else {  // (no word timestamps: the segment's words spread over its time)
      std::vector<std::string> ws;
      std::string cur;
      for (char c : g.text + " ") {
        if (isspace((unsigned char)c)) { if (!cur.empty()) ws.push_back(cur); cur.clear(); }
        else cur += c;
      }
      for (size_t i = 0; i < ws.size(); ++i) {
        Word w;
        w.text = ws[i];
        w.start = g.start + (g.end - g.start) * i / ws.size();
        w.end = g.start + (g.end - g.start) * (i + 1) / ws.size();
        w.p = 1;
        all.push_back(w);
      }
    }
    who.resize(all.size(), &g.speaker);
  }
  std::vector<Cue> cues;
  Cue cur;
  for (size_t i = 0; i < all.size(); ++i) {
    const Word& w = all[i];
    if (!cur.words.empty() && *who[i] != cur.speaker) { cues.push_back(cur); cur = Cue(); }
    if (!cur.words.empty()) {
      // (the cue's words with this one must still fit its lines as they will be laid out)
      std::vector<Word> with = cur.words;
      with.push_back(w);
      const bool long_text = line_breaks(with, 0, with.size(), o.max_chars, o.max_lines).empty();
      const bool long_time = w.end - cur.start > o.max_s;
      const bool pause = w.start - cur.words.back().end > o.pause_s;
      const char last = cur.words.back().text.empty() ? 0 : cur.words.back().text.back();
      const bool sentence = (last == '.' || last == '?' || last == '!') && cur.end - cur.start >= o.min_s;
      if (o.one_word || long_text || long_time || pause || sentence) { cues.push_back(cur); cur = Cue(); }
    }
    if (cur.words.empty()) {
      cur.start = w.start;
      cur.speaker = *who[i];
    }
    cur.words.push_back(w);
    cur.end = w.end;
  }
  if (!cur.words.empty()) cues.push_back(cur);
  for (Cue& c : cues) {  // lines of at most max_chars, two evened out
    c.first = line_breaks(c.words, 0, c.words.size(), o.max_chars, o.max_lines);
    if (c.first.empty()) c.first = {0};  // (one word longer than max_lines lines: alone)
    for (size_t l = 0; l < c.first.size(); ++l) {
      std::string line;
      for (size_t k = c.first[l]; k < (l + 1 < c.first.size() ? c.first[l + 1] : c.words.size()); ++k)
        line += (line.empty() ? "" : " ") + c.words[k].text;
      c.lines.push_back(line);
    }
    if (c.end <= c.start) c.end = c.start + 0.3;
  }
  // in time order, and no overlap within a speaker's cues (two speakers' cues may overlap: people talk at once, and
  // every format shows both). The tracks' segments come merged by their starts, so a short reply of the other track
  // sat between a long segment's cues: clamped to it, the earlier cue ended before it started.
  std::stable_sort(cues.begin(), cues.end(), [](const Cue& a, const Cue& b) { return a.start < b.start; });
  for (size_t i = 0; i < cues.size(); ++i)
    for (size_t j = i + 1; j < cues.size(); ++j)
      if (cues[j].speaker == cues[i].speaker) {
        if (cues[i].end > cues[j].start) cues[i].end = std::max(cues[j].start, cues[i].start + 0.01);
        break;
      }
  return cues;
}

// ---------------------------------------------------------------- the formats
inline std::string to_json(const Result& r) {
  std::string o = "{\"duration\": " + num(r.duration) + ", \"segments\": [";
  for (size_t i = 0; i < r.segments.size(); ++i) {
    const Segment& g = r.segments[i];
    o += (i ? ",\n" : "\n") + std::string(" {\"id\": ") + std::to_string(i) + ", \"start\": " + num(g.start) + ", \"end\": " +
         num(g.end) + ", \"lang\": " + json_str(g.lang) +
         (g.speaker.empty() ? std::string() : ", \"speaker\": " + json_str(g.speaker)) + ", \"text\": " + json_str(trim_copy(g.text)) +
         ", \"avg_logprob\": " + num(g.avg_logprob, 4) + ", \"no_speech_prob\": " + num(g.no_speech_prob, 4) +
         ", \"compression_ratio\": " + num(g.compression_ratio, 3) + ", \"temperature\": " + num(g.temperature, 1) + ", \"words\": [";
    for (size_t w = 0; w < g.words.size(); ++w)
      o += (w ? ", " : "") + std::string("{\"start\": ") + num(g.words[w].start) + ", \"end\": " + num(g.words[w].end) +
           ", \"word\": " + json_str(g.words[w].text) + ", \"p\": " + num(g.words[w].p, 3) + "}";
    o += "]}";
  }
  return o + "\n]}\n";
}
// the JSON tools/audio/transcribe.py (faster-whisper) wrote for the Remotion pipeline (edit.py, timeline.py)
inline std::string to_fw(const Result& r) {
  std::string o = "{\"duration\": " + num(r.duration) + ", \"segments\": [";
  for (size_t i = 0; i < r.segments.size(); ++i) {
    const Segment& g = r.segments[i];
    o += (i ? ",\n" : "\n") + std::string(" {\"start\": ") + num(g.start) + ", \"end\": " + num(g.end) + ", \"text\": " +
         json_str(trim_copy(g.text)) + ", \"lang\": " + json_str(g.lang) +
         (g.speaker.empty() ? std::string() : ", \"speaker\": " + json_str(g.speaker)) + ", \"words\": [";
    for (size_t w = 0; w < g.words.size(); ++w)
      o += (w ? ", " : "") + std::string("{\"start\": ") + num(g.words[w].start) + ", \"end\": " + num(g.words[w].end) +
           ", \"word\": " + json_str(g.words[w].text) + ", \"p\": " + num(g.words[w].p, 3) + "}";
    o += "]}";
  }
  return o + "\n]}\n";
}
// Remotion's Caption[] (@remotion/captions): one per word, text with its leading space
inline std::string to_captions(const Result& r) {
  std::string o = "[";
  bool first = true;
  for (const Segment& g : r.segments) {
    std::vector<Word> ws = g.words;
    if (ws.empty()) { Word w; w.text = trim_copy(g.text); w.start = g.start; w.end = g.end; w.p = 1; ws.push_back(w); }
    for (const Word& w : ws) {
      const long long s = std::llround(w.start * 1000), e = std::llround(w.end * 1000);
      o += (first ? "\n" : ",\n") + std::string(" {\"text\": ") + json_str((first ? "" : " ") + w.text) + ", \"startMs\": " +
           std::to_string(s) + ", \"endMs\": " + std::to_string(e) + ", \"timestampMs\": " + std::to_string((s + e) / 2) +
           ", \"confidence\": " + num(w.p, 3) + "}";
      first = false;
    }
  }
  return o + "\n]\n";
}
// OpenAI /v1/audio/transcriptions response_format=verbose_json
inline std::string to_verbose_json(const Result& r, const std::string& task) {
  std::string lang = r.segments.empty() ? "" : r.segments[0].lang;
  std::string o = "{\"task\": " + json_str(task) + ", \"language\": " + json_str(lang) + ", \"duration\": " + num(r.duration) +
                  ", \"text\": " + json_str(trim_copy(r.text())) + ", \"segments\": [";
  bool any_words = false;
  for (size_t i = 0; i < r.segments.size(); ++i) {
    const Segment& g = r.segments[i];
    any_words = any_words || !g.words.empty();
    o += (i ? ", " : "") + std::string("{\"id\": ") + std::to_string(i) + ", \"seek\": " +
         std::to_string((long long)std::llround(g.start * 100)) + ", \"start\": " + num(g.start) + ", \"end\": " + num(g.end) +
         ", \"text\": " + json_str(g.text) + ", \"tokens\": [";
    for (size_t t = 0; t < g.tokens.size(); ++t) o += (t ? ", " : "") + std::to_string(g.tokens[t]);
    o += "], \"temperature\": " + num(g.temperature, 1) + ", \"avg_logprob\": " + num(g.avg_logprob, 4) +
         ", \"compression_ratio\": " + num(g.compression_ratio, 3) + ", \"no_speech_prob\": " + num(g.no_speech_prob, 4) + "}";
  }
  o += "]";
  if (any_words) {
    o += ", \"words\": [";
    bool first = true;
    for (const Segment& g : r.segments)
      for (const Word& w : g.words) {
        o += (first ? "" : ", ") + std::string("{\"word\": ") + json_str(w.text) + ", \"start\": " + num(w.start) + ", \"end\": " + num(w.end) + "}";
        first = false;
      }
    o += "]";
  }
  return o + "}\n";
}
inline std::string to_openai_json(const Result& r) { return "{\"text\": " + json_str(trim_copy(r.text())) + "}\n"; }
inline std::string to_txt(const Result& r, bool times = false) {
  std::string o;
  const std::string* last = nullptr;
  for (const Segment& g : r.segments) {
    if (times) o += "[" + clock(g.start, '.', 3).substr(0, 8) + "] ";
    if (!g.speaker.empty() && (!last || *last != g.speaker)) o += g.speaker + ": ";  // (when the speaker changes)
    last = &g.speaker;
    o += trim_copy(g.text) + "\n";
  }
  return o;
}
inline std::string to_srt(const std::vector<Cue>& cues) {
  std::string o;
  for (size_t i = 0; i < cues.size(); ++i) {
    o += std::to_string(i + 1) + "\n" + clock(cues[i].start, ',', 3) + " --> " + clock(cues[i].end, ',', 3) + "\n";
    for (const std::string& l : cues[i].lines) o += l + "\n";
    o += "\n";
  }
  return o;
}
// WebVTT; word_timing: <hh:mm:ss.mmm> before each word after the first (karaoke-style highlighting), each later
// than the cue's start and the one before and earlier than its end, as WebVTT asks (a word that starts with the one
// before it, or past the end, gets none)
inline std::string to_vtt(const std::vector<Cue>& cues, bool word_timing) {
  std::string o = "WEBVTT\n\n";
  for (const Cue& c : cues) {
    o += clock(c.start, '.', 3) + " --> " + clock(c.end, '.', 3) + "\n";
    if (word_timing && c.words.size() > 1) {
      std::string text;
      // (the times as they are written: milliseconds)
      long long last = std::llround(c.start * 1000), end = std::llround(c.end * 1000);
      for (size_t w = 0; w < c.words.size(); ++w) {
        if (w) text += std::find(c.first.begin(), c.first.end(), w) != c.first.end() ? "\n" : " ";
        const long long t = std::llround(c.words[w].start * 1000);
        if (w && t > last && t < end) {
          text += "<" + clock(t / 1000.0, '.', 3) + ">";
          last = t;
        }
        text += xml(c.words[w].text);
      }
      o += (c.speaker.empty() ? "" : "<v " + xml(c.speaker) + ">") + text + "\n\n";
    } else {
      if (!c.speaker.empty()) o += "<v " + xml(c.speaker) + ">";
      for (const std::string& l : c.lines) o += xml(l) + "\n";
      o += "\n";
    }
  }
  return o;
}
inline std::string to_sbv(const std::vector<Cue>& cues) {
  std::string o;
  for (const Cue& c : cues) {
    o += clock(c.start, '.', 3, false) + "," + clock(c.end, '.', 3, false) + "\n";
    for (const std::string& l : c.lines) o += l + "\n";
    o += "\n";
  }
  return o;
}
// YouTube SRV3 (timed text format 3): each cue a <p>, each word an <s> with its offset (word-by-word display)
inline std::string to_srv3(const std::vector<Cue>& cues) {
  std::string o = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<timedtext format=\"3\">\n<body>\n";
  for (const Cue& c : cues) {
    const long long t = std::llround(c.start * 1000), d = std::max(1LL, std::llround((c.end - c.start) * 1000));
    o += "<p t=\"" + std::to_string(t) + "\" d=\"" + std::to_string(d) + "\">";
    for (size_t w = 0; w < c.words.size(); ++w) {
      const long long off = std::max(0LL, std::llround(c.words[w].start * 1000) - t);
      o += "<s" + (w ? " t=\"" + std::to_string(off) + "\"" : std::string()) + ">" + xml((w ? " " : "") + c.words[w].text) + "</s>";
    }
    o += "</p>\n";
  }
  return o + "</body>\n</timedtext>\n";
}
// ASS (Aegisub, mpv, ffmpeg burn-in): one Dialogue per cue, \k karaoke timing per word (centiseconds)
inline std::string to_ass(const std::vector<Cue>& cues) {
  std::string o =
      "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n\n"
      "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
      "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
      "Style: Default,Arial,64,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,60,1\n\n"
      "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";
  for (const Cue& c : cues) {
    std::string text;
    for (size_t w = 0; w < c.words.size(); ++w) {
      const double nxt = w + 1 < c.words.size() ? c.words[w + 1].start : c.end;
      const long long k = std::max(1LL, std::llround((nxt - c.words[w].start) * 100));
      std::string t = c.words[w].text;
      for (char& ch : t) if (ch == '{' || ch == '}') ch = ' ';
      text += "{\\k" + std::to_string(k) + "}" + t + (w + 1 < c.words.size() ? " " : "");
    }
    std::string name = c.speaker;  // (the Name field: no commas)
    for (char& ch : name) if (ch == ',') ch = ' ';
    o += "Dialogue: 0," + clock(c.start, '.', 2, false) + "," + clock(c.end, '.', 2, false) + ",Default," + name + ",0,0,0,," + text + "\n";
  }
  return o;
}
inline std::string to_ttml(const std::vector<Cue>& cues, const std::string& lang) {
  std::string o = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<tt xmlns=\"http://www.w3.org/ns/ttml\" xml:lang=\"" + xml(lang) +
                  "\">\n<body>\n<div>\n";
  for (const Cue& c : cues) {
    o += "<p begin=\"" + clock(c.start, '.', 3) + "\" end=\"" + clock(c.end, '.', 3) + "\">";
    for (size_t i = 0; i < c.lines.size(); ++i) o += (i ? "<br/>" : "") + xml(c.lines[i]);
    o += "</p>\n";
  }
  return o + "</div>\n</body>\n</tt>\n";
}
inline std::string to_md(const Result& r, const std::string& title) {
  std::string o = "# " + title + "\n\n";
  for (const Segment& g : r.segments)
    o += "**[" + clock(g.start, '.', 3).substr(0, 8) + "]" + (g.speaker.empty() ? "" : " " + g.speaker + ":") + "** " +
         trim_copy(g.text) + "\n\n";
  return o;
}
inline std::string to_table(const Result& r, char sep) {
  auto cell = [&](const std::string& s) {
    if (sep == '\t') { std::string t = s; for (char& c : t) if (c == '\t' || c == '\n') c = ' '; return t; }
    std::string t = "\"";
    for (char c : s) { if (c == '"') t += '"'; t += c; }
    return t + "\"";
  };
  const bool who = std::any_of(r.segments.begin(), r.segments.end(), [](const Segment& g) { return !g.speaker.empty(); });
  std::string o = std::string("start") + sep + "end" + sep + "lang" + sep + (who ? std::string("speaker") + sep : "") + "text\n";
  for (const Segment& g : r.segments)
    o += num(g.start) + sep + num(g.end) + sep + g.lang + sep + (who ? cell(g.speaker) + sep : "") + cell(trim_copy(g.text)) + "\n";
  return o;
}

// every format by its name; ext: the file extension
struct Format {
  const char* name;
  const char* ext;
};
inline const std::vector<Format>& formats() {
  static const std::vector<Format> f = {
      {"json", ".json"}, {"fw", ".fw.json"}, {"captions", ".captions.json"}, {"verbose_json", ".openai.json"},
      {"srt", ".srt"}, {"vtt", ".vtt"}, {"sbv", ".sbv"}, {"srv3", ".srv3"}, {"ass", ".ass"}, {"ttml", ".ttml"},
      {"txt", ".txt"}, {"md", ".md"}, {"csv", ".csv"}, {"tsv", ".tsv"}};
  return f;
}
inline std::string render(const std::string& fmt, const Result& r, const CueOptions& co, const std::string& title,
                          const std::string& task = "transcribe") {
  const std::string lang = r.segments.empty() ? "und" : r.segments[0].lang;
  if (fmt == "json") return to_json(r);
  if (fmt == "fw") return to_fw(r);
  if (fmt == "captions") return to_captions(r);
  if (fmt == "verbose_json") return to_verbose_json(r, task);
  if (fmt == "openai_json") return to_openai_json(r);
  if (fmt == "txt" || fmt == "text") return to_txt(r);
  if (fmt == "md") return to_md(r, title);
  if (fmt == "csv") return to_table(r, ',');
  if (fmt == "tsv") return to_table(r, '\t');
  const std::vector<Cue> cues = make_cues(r, co);
  if (fmt == "srt") return to_srt(cues);
  if (fmt == "vtt") return to_vtt(cues, true);
  if (fmt == "sbv") return to_sbv(cues);
  if (fmt == "srv3") return to_srv3(cues);
  if (fmt == "ass") return to_ass(cues);
  if (fmt == "ttml") return to_ttml(cues, lang);
  throw std::runtime_error("unknown transcript format '" + fmt + "'");
}

}  // namespace whisper
