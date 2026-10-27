// Reading the documents users upload (DOCX, PDF, text, Markdown, CSV ...) as paragraphs with their page and heading
// level, for the web UI's file tools: a long document is not put into the prompt whole (100 pages are ~60K tokens:
// minutes of prompt processing), the model gets its outline and reads the sections it needs, or searches it.
//   read(name, bytes)   -> Doc: paragraphs (text, page, heading level), page count, a note when nothing was read
//   Index(doc)          -> its outline (the headings, or the pages when it has none), chunks of ~2,400 characters along
//                          paragraphs, and a BM25 index over them: search(query) gives the chunks that fit best
// DOCX: word/document.xml (headings from styles.xml's "heading N" / outline levels, tables as "a | b | c" rows, pages
// from Word's rendered page breaks). PDF: the content streams' text (fonts with ToUnicode maps or simple encodings,
// form XObjects), lines grouped into paragraphs by their spacing, headings by font size. Not read: scans without a
// text layer (that needs OCR), encrypted PDFs, legacy binary .doc.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "inflate.h"

namespace rdoc {

struct Para {
  std::string text;
  int page = 1;
  int level = 0;  // (a heading: 1..6; body text: 0)
};
struct Doc {
  std::vector<Para> paras;
  int pages = 0;  // (0: the format has no pages, e.g. a text file)
  std::string note;  // (why little or nothing was read)
  size_t chars() const {
    size_t n = 0;
    for (const Para& p : paras) n += p.text.size() + 2;
    return n;
  }
};

inline void put_utf8(std::string& o, uint32_t c) {
  if (c < 0x80) o += (char)c;
  else if (c < 0x800) { o += (char)(0xC0 | c >> 6); o += (char)(0x80 | (c & 63)); }
  else if (c < 0x10000) { o += (char)(0xE0 | c >> 12); o += (char)(0x80 | (c >> 6 & 63)); o += (char)(0x80 | (c & 63)); }
  else if (c < 0x110000) { o += (char)(0xF0 | c >> 18); o += (char)(0x80 | (c >> 12 & 63)); o += (char)(0x80 | (c >> 6 & 63)); o += (char)(0x80 | (c & 63)); }
}
// spaces squeezed, trimmed
inline std::string tidy(const std::string& s) {
  std::string o;
  for (char c : s) {
    const char d = (c == '\t' || c == '\r' || c == '\n') ? ' ' : c;
    if (d == ' ' && (o.empty() || o.back() == ' ')) continue;
    o += d;
  }
  while (!o.empty() && o.back() == ' ') o.pop_back();
  return o;
}
inline bool valid_utf8(const std::string& s) {
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = (unsigned char)s[i];
    const int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (!n || i + n > s.size()) return false;
    for (int k = 1; k < n; ++k)
      if (((unsigned char)s[i + k] >> 6) != 2) return false;
    i += n;
  }
  return true;
}
inline uint32_t cp1252(unsigned char b) {
  static const uint16_t hi[32] = {0x20AC, 0, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0, 0x017D, 0,
                                  0, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0, 0x017E, 0x0178};
  return b >= 0x80 && b < 0xA0 ? (hi[b - 0x80] ? hi[b - 0x80] : '?') : b;
}

// ---------------------------------------------------------------- XML (DOCX)
inline std::string xml_text(const std::string& s) {  // (entities)
  std::string o;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '&') { o += s[i]; continue; }
    const size_t e = s.find(';', i);
    if (e == std::string::npos || e - i > 10) { o += '&'; continue; }
    const std::string n = s.substr(i + 1, e - i - 1);
    if (n == "amp") o += '&';
    else if (n == "lt") o += '<';
    else if (n == "gt") o += '>';
    else if (n == "quot") o += '"';
    else if (n == "apos") o += '\'';
    else if (!n.empty() && n[0] == '#') put_utf8(o, (uint32_t)strtoul(n.c_str() + 1 + (n.size() > 1 && n[1] == 'x'), nullptr, n.size() > 1 && n[1] == 'x' ? 16 : 10));
    else { o += '&'; continue; }
    i = e;
  }
  return o;
}
inline std::string xml_attr(const std::string& tag, const char* name) {
  const std::string k = std::string(" ") + name + "=\"";
  const size_t a = tag.find(k);
  if (a == std::string::npos) return "";
  const size_t b = tag.find('"', a + k.size());
  return b == std::string::npos ? "" : xml_text(tag.substr(a + k.size(), b - a - k.size()));
}
// for each tag: f(name, whole tag, the text after it up to the next tag)
inline void xml_walk(const std::string& x, const std::function<void(const std::string&, const std::string&, size_t, size_t)>& f) {
  for (size_t i = x.find('<'); i != std::string::npos;) {
    const size_t e = x.find('>', i);
    if (e == std::string::npos) break;
    const size_t ne = x.find_first_of(" />\t\r\n", i + (i + 1 < x.size() && x[i + 1] == '/' ? 2 : 1));
    const std::string name = x.substr(i + 1, std::min(ne, e) - i - 1);
    const size_t next = x.find('<', e);
    f(name, x.substr(i, e - i + 1), e + 1, next == std::string::npos ? x.size() : next);
    i = next;
  }
}

inline Doc read_docx(const std::string& zip) {
  Doc d;
  std::string x, st;
  if (!infl::zip_entry(zip, "word/document.xml", x, 64u << 20)) {
    d.note = "not a readable DOCX (no word/document.xml)";
    return d;
  }
  // style id -> heading level ("heading 1".."heading 6", "Title", or an outline level)
  std::map<std::string, int> lvl;
  if (infl::zip_entry(zip, "word/styles.xml", st, 16u << 20)) {
    std::string id;
    xml_walk(st, [&](const std::string& n, const std::string& tag, size_t, size_t) {
      if (n == "w:style") id = xml_attr(tag, "w:styleId");
      else if (n == "w:name" && !id.empty()) {
        std::string v = xml_attr(tag, "w:val");
        for (char& c : v) c = (char)tolower((unsigned char)c);
        if (v.rfind("heading ", 0) == 0 && v.size() == 9 && v[8] >= '1' && v[8] <= '6') lvl[id] = v[8] - '0';
        else if (v == "title") lvl[id] = 1;
      } else if (n == "w:outlineLvl" && !id.empty() && !lvl.count(id)) {
        const int o = atoi(xml_attr(tag, "w:val").c_str());
        if (o >= 0 && o < 6) lvl[id] = o + 1;
      }
    });
  }
  const bool rendered = x.find("<w:lastRenderedPageBreak") != std::string::npos;
  int page = 1, level = 0, tbl = 0;
  bool in_t = false, list = false;
  std::string p, cell, row;
  auto end_para = [&] {
    std::string t = tidy(p);
    p.clear();
    if (tbl > 0) {
      if (!t.empty()) cell += (cell.empty() ? "" : " ") + t;
    } else if (!t.empty()) d.paras.push_back({(list && !level ? "- " : "") + t, page, level});
    level = 0;
    list = false;
  };
  xml_walk(x, [&](const std::string& n, const std::string& tag, size_t a, size_t b) {
    if (n == "w:t") {
      in_t = tag.back() != '/' || tag[tag.size() - 2] != '/';
      if (in_t) p += xml_text(x.substr(a, b - a));
    } else if (n == "/w:t") in_t = false;
    else if (n == "w:tab") p += '\t';
    else if (n == "w:br" || n == "w:cr") {
      if (xml_attr(tag, "w:type") == "page") { if (!rendered) ++page; }
      else p += ' ';
    } else if (n == "w:lastRenderedPageBreak") ++page;
    else if (n == "w:pStyle") {
      auto it = lvl.find(xml_attr(tag, "w:val"));
      if (it != lvl.end()) level = it->second;
    } else if (n == "w:outlineLvl") {
      const int o = atoi(xml_attr(tag, "w:val").c_str());
      if (o >= 0 && o < 6) level = o + 1;
    } else if (n == "w:numPr") list = true;
    else if (n == "/w:p") end_para();
    else if (n == "w:tbl") ++tbl;
    else if (n == "/w:tbl") tbl = std::max(0, tbl - 1);
    else if (n == "/w:tc" && tbl > 0) { row += (row.empty() ? "" : " | ") + cell; cell.clear(); }
    else if (n == "/w:tr" && tbl > 0) {
      if (tbl == 1 && !tidy(row).empty()) d.paras.push_back({tidy(row), page, 0});
      else if (tbl > 1) cell += (cell.empty() ? "" : " ") + row;  // (a nested table: inside its cell)
      row.clear();
    }
  });
  d.pages = page;
  // Word marks the page breaks it rendered, often only those on screen: when docProps/app.xml says the document has
  // clearly more pages, the pages are estimated from the text's position instead (between the breaks it marked)
  std::string app;
  if (infl::zip_entry(zip, "docProps/app.xml", app, 1u << 20)) {
    const size_t a = app.find("<Pages>");
    const int pages = a == std::string::npos ? 0 : atoi(app.c_str() + a + 7);
    if (pages > page * 3 / 2 + 1 && pages < 100000) {
      const double total = (double)std::max<size_t>(1, d.chars());
      double acc = 0;
      for (Para& q : d.paras) {
        q.page = std::max(q.page, 1 + (int)(acc / total * pages));
        acc += q.text.size() + 2;
      }
      d.pages = pages;
    }
  }
  if (d.paras.empty()) d.note = "the DOCX has no text";
  return d;
}

// ---------------------------------------------------------------- PDF
namespace pdf {

struct Obj;
using P = std::shared_ptr<Obj>;
struct Obj {
  enum T { Null, Bool, Num, Str, Name, Arr, Dict, Ref } t = Null;
  double n = 0;
  std::string s;  // (string bytes, name)
  std::vector<P> a;
  std::vector<std::pair<std::string, P>> d;
  int ref = 0;
  std::string stream;  // (a stream object's raw bytes)
  bool has_stream = false;
  const P get(const std::string& k) const {
    for (const auto& kv : d)
      if (kv.first == k) return kv.second;
    return nullptr;
  }
};

struct Lexer {
  const std::string& s;
  size_t i;
  static bool ws(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == 0; }
  static bool delim(char c) { return strchr("()<>[]{}/%", c) != nullptr; }
  void skip() {
    while (i < s.size()) {
      if (ws(s[i])) ++i;
      else if (s[i] == '%') { while (i < s.size() && s[i] != '\n' && s[i] != '\r') ++i; }
      else break;
    }
  }
  std::string literal() {  // ( ... ) with escapes, at '('
    std::string o;
    int depth = 1;
    ++i;
    while (i < s.size() && depth > 0) {
      char c = s[i++];
      if (c == '\\' && i < s.size()) {
        c = s[i++];
        switch (c) {
          case 'n': o += '\n'; break;
          case 'r': o += '\r'; break;
          case 't': o += '\t'; break;
          case 'b': o += '\b'; break;
          case 'f': o += '\f'; break;
          case '\r': if (i < s.size() && s[i] == '\n') ++i; break;
          case '\n': break;
          default:
            if (c >= '0' && c <= '7') {
              int v = c - '0';
              for (int k = 0; k < 2 && i < s.size() && s[i] >= '0' && s[i] <= '7'; ++k) v = v * 8 + (s[i++] - '0');
              o += (char)v;
            } else o += c;
        }
      } else {
        if (c == '(') ++depth;
        else if (c == ')' && --depth == 0) break;
        o += c;
      }
    }
    return o;
  }
  std::string hex() {  // < ... >, at '<'
    std::string o;
    ++i;
    int hv = -1;
    while (i < s.size() && s[i] != '>') {
      const char c = s[i++];
      const int v = isdigit((unsigned char)c) ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
      if (v < 0) continue;
      if (hv < 0) hv = v;
      else { o += (char)(hv * 16 + v); hv = -1; }
    }
    if (hv >= 0) o += (char)(hv * 16);
    ++i;
    return o;
  }
  std::string name() {  // at '/'
    std::string o;
    ++i;
    while (i < s.size() && !ws(s[i]) && !delim(s[i])) {
      if (s[i] == '#' && i + 2 < s.size() && isxdigit((unsigned char)s[i + 1]) && isxdigit((unsigned char)s[i + 2])) {
        o += (char)strtol(s.substr(i + 1, 2).c_str(), nullptr, 16);
        i += 3;
      } else o += s[i++];
    }
    return o;
  }
  std::string word() {
    const size_t b = i;
    while (i < s.size() && !ws(s[i]) && !delim(s[i])) ++i;
    return s.substr(b, i - b);
  }
  // one object (refs "n g R" folded in); nullptr at the end or at an operator (then op holds it)
  P value(std::string* op = nullptr, int depth = 0) {
    skip();
    // (stray closing delimiters skipped in a loop: a file of "]]]]..." must not recurse once per character)
    while (i < s.size() && (s[i] == ']' || s[i] == ')' || s[i] == '{' || s[i] == '}' || (s[i] == '>' && (i + 1 >= s.size() || s[i + 1] != '>')))) {
      ++i;
      skip();
    }
    if (i >= s.size() || depth > 64) return nullptr;
    const char c = s[i];
    auto o = std::make_shared<Obj>();
    if (c == '(') { o->t = Obj::Str; o->s = literal(); return o; }
    if (c == '<' && i + 1 < s.size() && s[i + 1] == '<') {
      i += 2;
      o->t = Obj::Dict;
      while (true) {
        skip();
        if (i >= s.size()) return o;
        if (s[i] == '>' && i + 1 < s.size() && s[i + 1] == '>') { i += 2; return o; }
        if (s[i] != '/') { ++i; continue; }
        const std::string k = name();
        P v = value(nullptr, depth + 1);
        if (!v) return o;
        o->d.push_back({k, v});
      }
    }
    if (c == '<') { o->t = Obj::Str; o->s = hex(); return o; }
    if (c == '[') {
      ++i;
      o->t = Obj::Arr;
      while (true) {
        skip();
        if (i >= s.size()) return o;
        if (s[i] == ']') { ++i; return o; }
        P v = value(nullptr, depth + 1);
        if (!v) { ++i; continue; }
        o->a.push_back(v);
      }
    }
    if (c == '/') { o->t = Obj::Name; o->s = name(); return o; }
    if (c == ']' || c == '>' || c == ')' || c == '{' || c == '}') { ++i; return nullptr; }  // (a stray ">>": skipped, no recursion)
    const std::string w = word();
    if (w.empty()) { ++i; return nullptr; }
    if (isdigit((unsigned char)w[0]) || w[0] == '-' || w[0] == '+' || w[0] == '.') {
      o->t = Obj::Num;
      o->n = atof(w.c_str());
      // "n g R"?
      const size_t save = i;
      skip();
      const std::string g = word();
      if (!g.empty() && isdigit((unsigned char)g[0])) {
        skip();
        if (i < s.size() && s[i] == 'R' && (i + 1 >= s.size() || ws(s[i + 1]) || delim(s[i + 1]))) {
          ++i;
          o->t = Obj::Ref;
          o->ref = (int)o->n;
          return o;
        }
      }
      i = save;
      return o;
    }
    if (w == "true" || w == "false") { o->t = Obj::Bool; o->n = w == "true"; return o; }
    if (w == "null") return o;
    if (op) *op = w;
    return nullptr;
  }
};

class File {
 public:
  explicit File(const std::string& data) : s_(data) { load(); }
  bool encrypted = false;

  P resolve(P o, int depth = 0) const {
    while (o && o->t == Obj::Ref && depth++ < 32) {
      auto it = objs_.find(o->ref);
      o = it == objs_.end() ? nullptr : it->second;
    }
    return o;
  }
  P get(const P& dict, const std::string& k) const { return dict ? resolve(dict->get(k)) : nullptr; }
  // a stream's decoded bytes (FlateDecode; no filter); "" for others
  std::string decode(const P& o) const {
    if (!o || !o->has_stream) return "";
    P f = get(o, "Filter");
    std::vector<std::string> fs;
    if (f && f->t == Obj::Name) fs.push_back(f->s);
    else if (f && f->t == Obj::Arr)
      for (auto& x : f->a)
        if (P r = resolve(x); r && r->t == Obj::Name) fs.push_back(r->s);
    std::string cur = o->stream;
    for (const std::string& name : fs) {
      if (name == "FlateDecode" || name == "Fl") {
        std::string out;
        infl::zlib(cur, out, std::min<size_t>(128u << 20, budget_));  // (a damaged stream: what decoded)
        budget_ -= std::min(budget_, out.size());  // (all streams of one file together: 512 MB at most)
        P parms = get(o, "DecodeParms");
        if (parms && parms->t == Obj::Arr && !parms->a.empty()) parms = resolve(parms->a[0]);
        const P pr = get(parms, "Predictor"), cols = get(parms, "Columns");
        if (pr && pr->t == Obj::Num && pr->n >= 10) out = png_unpredict(out, cols && cols->t == Obj::Num ? (int)cols->n : 1);
        cur.swap(out);
      } else return "";  // (DCT, JBIG2, LZW ...: not text)
    }
    return cur;
  }
  // the pages in order: each with its (inherited) resources
  std::vector<std::pair<P, P>> pages() const {
    std::vector<std::pair<P, P>> v;
    std::set<const Obj*> seen;
    std::function<void(P, P, int)> walk = [&](P node, P res, int depth) {
      node = resolve(node);
      if (!node || depth > 64 || seen.count(node.get())) return;
      seen.insert(node.get());
      if (P r = get(node, "Resources")) res = r;
      P kids = get(node, "Kids");
      P type = get(node, "Type");
      if (kids && kids->t == Obj::Arr) {
        for (auto& k : kids->a) walk(k, res, depth + 1);
      } else if (!type || type->s == "Page") v.push_back({node, res});
    };
    if (P root = get(trailer_root(), "Pages")) walk(root, nullptr, 0);
    if (v.empty())  // (no usable page tree: the page objects in file order)
      for (const auto& [num, o] : objs_)
        if (P t = get(o, "Type"); t && t->t == Obj::Name && t->s == "Page") v.push_back({o, get(o, "Resources")});
    return v;
  }

 private:
  const std::string& s_;
  std::map<int, P> objs_;
  mutable size_t budget_ = 512u << 20;

  static std::string png_unpredict(const std::string& in, int cols) {
    std::string out;
    const int rowlen = std::max(1, cols) + 1;
    std::string prev((size_t)cols, '\0');
    for (size_t r = 0; r + rowlen <= in.size(); r += rowlen) {
      const int ft = (unsigned char)in[r];
      std::string row = in.substr(r + 1, (size_t)cols);
      for (int k = 0; k < cols; ++k) {
        const int a = k ? (unsigned char)row[k - 1] : 0, b = (unsigned char)prev[k], c = k ? (unsigned char)prev[k - 1] : 0;
        int x = (unsigned char)row[k];
        if (ft == 1) x += a;
        else if (ft == 2) x += b;
        else if (ft == 3) x += (a + b) / 2;
        else if (ft == 4) {
          const int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
          x += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
        }
        row[k] = (char)x;
      }
      out += row;
      prev = row;
    }
    return out;
  }
  P trailer_root() const {
    for (const auto& [num, o] : objs_)
      if (P t = get(o, "Type"); t && t->t == Obj::Name && t->s == "Catalog") return o;
    return nullptr;
  }
  // every "n g obj ... endobj" in the file (the last one of a number wins: incremental updates), and the objects
  // inside object streams. The cross-reference table is not needed (and often wrong in damaged files).
  void load() {
    std::vector<P> objstms;
    for (size_t i = 0; (i = s_.find(" obj", i)) != std::string::npos; i += 4) {
      // back over "n g"
      size_t b = i;
      while (b > 0 && isdigit((unsigned char)s_[b - 1])) --b;
      if (b == i || b == 0 || s_[b - 1] != ' ') continue;
      size_t a = b - 1;
      while (a > 0 && isdigit((unsigned char)s_[a - 1])) --a;
      if (a == b - 1 || (a > 0 && !Lexer::ws(s_[a - 1]) && s_[a - 1] != '>')) continue;
      const int num = atoi(s_.c_str() + a);
      Lexer lx{s_, i + 4};
      P o = lx.value();
      if (!o) continue;
      lx.skip();
      if (o->t == Obj::Dict && s_.compare(lx.i, 6, "stream") == 0) {
        size_t st = lx.i + 6;
        if (st < s_.size() && s_[st] == '\r') ++st;
        if (st < s_.size() && s_[st] == '\n') ++st;
        P len = o->get("Length");
        size_t n = len && len->t == Obj::Num ? (size_t)len->n : std::string::npos;
        if (n == std::string::npos || st + n > s_.size() || s_.find("endstream", st + n) > st + n + 4) {
          const size_t e = s_.find("endstream", st);  // (an indirect or wrong /Length)
          n = (e == std::string::npos ? s_.size() : e) - st;
          while (n > 0 && (s_[st + n - 1] == '\n' || s_[st + n - 1] == '\r')) --n;
        }
        o->stream = s_.substr(st, n);
        o->has_stream = true;
        i = st + n;
        if (P t = o->get("Type"); t && t->t == Obj::Name && t->s == "ObjStm") objstms.push_back(o);
      }
      objs_[num] = o;
    }
    for (const P& os : objstms) {  // (PDF 1.5: small objects packed into a compressed stream)
      const std::string body = decode(os);
      P np = get(os, "N"), fp = get(os, "First");
      if (!np || !fp) continue;
      Lexer hx{body, 0};
      std::vector<std::pair<int, size_t>> idx;
      for (int k = 0; k < (int)np->n; ++k) {
        P a = hx.value(), b = hx.value();
        if (!a || !b) break;
        idx.push_back({(int)a->n, (size_t)b->n});
      }
      for (const auto& [num, off] : idx) {
        if (objs_.count(num) && !objs_[num]->has_stream && objs_[num]->t != Obj::Null) continue;
        Lexer lx{body, (size_t)fp->n + off};
        if (P o = lx.value()) objs_[num] = o;
      }
    }
    for (const auto& [num, o] : objs_)
      if (o->t == Obj::Dict && o->get("Encrypt")) encrypted = true;  // (an xref stream's trailer)
    encrypted = encrypted || s_.find("/Encrypt") != std::string::npos;
  }
};

// glyph names of simple fonts' /Differences that matter for text
inline uint32_t glyph_unicode(const std::string& n) {
  if (n.size() == 1) return (unsigned char)n[0];
  if (n.rfind("uni", 0) == 0 && n.size() >= 7) return (uint32_t)strtoul(n.substr(3, 4).c_str(), nullptr, 16);
  static const std::map<std::string, uint32_t> m = {
      {"space", ' '}, {"exclam", '!'}, {"quotedbl", '"'}, {"numbersign", '#'}, {"dollar", '$'}, {"percent", '%'},
      {"ampersand", '&'}, {"quotesingle", '\''}, {"parenleft", '('}, {"parenright", ')'}, {"asterisk", '*'}, {"plus", '+'},
      {"comma", ','}, {"hyphen", '-'}, {"period", '.'}, {"slash", '/'}, {"zero", '0'}, {"one", '1'}, {"two", '2'},
      {"three", '3'}, {"four", '4'}, {"five", '5'}, {"six", '6'}, {"seven", '7'}, {"eight", '8'}, {"nine", '9'},
      {"colon", ':'}, {"semicolon", ';'}, {"less", '<'}, {"equal", '='}, {"greater", '>'}, {"question", '?'}, {"at", '@'},
      {"bracketleft", '['}, {"backslash", '\\'}, {"bracketright", ']'}, {"underscore", '_'}, {"quoteleft", 0x2018},
      {"quoteright", 0x2019}, {"quotedblleft", 0x201C}, {"quotedblright", 0x201D}, {"endash", 0x2013}, {"emdash", 0x2014},
      {"bullet", 0x2022}, {"fi", 0xFB01}, {"fl", 0xFB02}, {"ellipsis", 0x2026}, {"eacute", 0xE9}, {"egrave", 0xE8},
      {"agrave", 0xE0}, {"udieresis", 0xFC}, {"odieresis", 0xF6}, {"adieresis", 0xE4}, {"ccedilla", 0xE7}, {"degree", 0xB0},
      {"copyright", 0xA9}, {"registered", 0xAE}, {"section", 0xA7}, {"Euro", 0x20AC}, {"nbspace", 0xA0}, {"minus", 0x2212}};
  auto it = m.find(n);
  return it == m.end() ? 0 : it->second;
}

struct Font {
  int bytes = 1;  // (code length: Type0 fonts 2)
  std::unordered_map<uint32_t, std::string> map;  // (code -> UTF-8, from ToUnicode / Differences)
  bool simple_ok = true;  // (a simple font without a map: bytes as Windows-1252)
  bool bold = false;
  std::unordered_map<uint32_t, double> w;  // (code -> advance in 1/1000 em: /Widths, /W)
  double dw = 500;                         // (a code without a width)
  uint32_t code_at(const std::string& s, size_t i) const {
    uint32_t c = 0;
    for (int k = 0; k < bytes; ++k) c = c << 8 | (unsigned char)s[i + k];
    return c;
  }
  std::string text(const std::string& codes) const {
    std::string o;
    for (size_t i = 0; i + bytes <= codes.size(); i += bytes) {
      const uint32_t c = code_at(codes, i);
      auto it = map.find(c);
      if (it != map.end()) o += it->second;
      else if (bytes == 1 && simple_ok) put_utf8(o, cp1252((unsigned char)c));
    }
    return o;
  }
  // the string's advance in text space (font size 1), with character and word spacing
  double advance(const std::string& codes, double size, double tc, double tw) const {
    double a = 0;
    for (size_t i = 0; i + bytes <= codes.size(); i += bytes) {
      const uint32_t c = code_at(codes, i);
      auto it = w.find(c);
      a += (it == w.end() ? dw : it->second) / 1000.0 * size + tc + (bytes == 1 && c == 32 ? tw : 0);
    }
    return a;
  }
};

inline void parse_cmap(const std::string& cm, Font& f) {
  Lexer lx{cm, 0};
  std::vector<P> ops;
  auto u16 = [](const std::string& s) {  // (UTF-16BE -> UTF-8)
    std::string o;
    for (size_t i = 0; i + 1 < s.size(); i += 2) {
      uint32_t c = (unsigned char)s[i] << 8 | (unsigned char)s[i + 1];
      if (c >= 0xD800 && c < 0xDC00 && i + 3 < s.size()) {
        const uint32_t lo = (unsigned char)s[i + 2] << 8 | (unsigned char)s[i + 3];
        c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
        i += 2;
      }
      put_utf8(o, c);
    }
    if (s.size() == 1) put_utf8(o, (unsigned char)s[0]);
    return o;
  };
  auto code = [](const std::string& s) {
    uint32_t c = 0;
    for (unsigned char b : s) c = c << 8 | b;
    return c;
  };
  while (lx.i < cm.size()) {
    std::string op;
    P v = lx.value(&op);
    if (v) { ops.push_back(v); continue; }
    if (op.empty()) { if (lx.i >= cm.size()) break; continue; }
    if (op == "endcodespacerange" && ops.size() >= 1 && ops[0]->t == Obj::Str) f.bytes = std::clamp((int)ops[0]->s.size(), 1, 4);
    else if (op == "endbfchar")
      for (size_t k = 0; k + 1 < ops.size(); k += 2) {
        if (ops[k]->t == Obj::Str && ops[k + 1]->t == Obj::Str) f.map[code(ops[k]->s)] = u16(ops[k + 1]->s);
      }
    else if (op == "endbfrange")
      for (size_t k = 0; k + 2 < ops.size(); k += 3) {
        if (ops[k]->t != Obj::Str || ops[k + 1]->t != Obj::Str) continue;
        const uint32_t lo = code(ops[k]->s), hi = code(ops[k + 1]->s);
        if (hi < lo || hi - lo > 65535) continue;
        if (ops[k + 2]->t == Obj::Arr) {
          for (uint32_t c = lo; c <= hi && c - lo < ops[k + 2]->a.size(); ++c) f.map[c] = u16(ops[k + 2]->a[c - lo]->s);
        } else if (ops[k + 2]->t == Obj::Str) {
          std::string dst = ops[k + 2]->s;
          for (uint32_t c = lo; c <= hi; ++c) {
            f.map[c] = u16(dst);
            if (!dst.empty()) {  // (the last byte counts up)
              size_t j = dst.size() - 1;
              while (true) {
                if (++dst[j] != 0 || j == 0) break;
                --j;
              }
            }
          }
        }
      }
    ops.clear();
  }
}

struct Line {
  double y, size;
  std::string text;
  int page;
  bool bold;     // (nearly all of it in a bold font)
  bool columns;  // (pieces far apart on the baseline: a table row)
};

// a page's text as lines: the content stream run with its text state (fonts, matrices, glyph advances), a new line
// where the baseline moves, a space where the next piece starts clearly after the previous one ended
class Reader {
 public:
  explicit Reader(const File& f) : f_(f) {}
  std::vector<Line> lines;

  void page(const P& pg, const P& res, int num) {
    page_ = num;
    cur_y_ = 1e9;
    std::string content;
    P c = f_.get(pg, "Contents");
    if (c && c->t == Obj::Arr) {
      for (auto& x : c->a) content += f_.decode(f_.resolve(x)) + "\n";
    } else content = f_.decode(c);
    run(content, res, 0, {1, 0, 0, 1, 0, 0});
    flush();
  }

 private:
  using M = std::array<double, 6>;
  const File& f_;
  int page_ = 1;
  std::map<const Obj*, Font> fonts_;
  std::string cur_;
  double cur_y_ = 1e9, cur_size_ = 0, last_x_ = 0;
  size_t bold_chars_ = 0, chars_ = 0;
  bool columns_ = false;
  int steps_ = 0;  // (operators run on this document: a cap against pathological files)

  static M mul(const M& a, const M& b) {
    return {a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5]};
  }
  void flush() {
    std::string t = tidy(cur_);
    while (t.size() >= 2 && t.compare(t.size() - 2, 2, " |") == 0) t = tidy(t.substr(0, t.size() - 2));
    if (!t.empty()) lines.push_back({cur_y_, cur_size_, t, page_, bold_chars_ * 10 >= chars_ * 9 && chars_ > 0, columns_});
    cur_.clear();
    bold_chars_ = chars_ = 0;
    columns_ = false;
  }
  const Font& font(const P& fo) {
    auto it = fonts_.find(fo.get());
    if (it != fonts_.end()) return it->second;
    Font f;
    P sub = f_.get(fo, "Subtype");
    if (sub && sub->s == "Type0") {
      f.bytes = 2;
      f.simple_ok = false;
      f.dw = 1000;
      P desc = f_.get(fo, "DescendantFonts");
      if (desc && desc->t == Obj::Arr && !desc->a.empty()) {
        P cid = f_.resolve(desc->a[0]);
        if (P dw = f_.get(cid, "DW"); dw && dw->t == Obj::Num) f.dw = dw->n;
        if (P W = f_.get(cid, "W"); W && W->t == Obj::Arr)  // ([c [w1 w2 ...]] or [c1 c2 w])
          for (size_t k = 0; k + 1 < W->a.size();) {
            P c0 = f_.resolve(W->a[k]), n1 = f_.resolve(W->a[k + 1]);
            if (!c0 || !n1) break;
            if (n1->t == Obj::Arr) {
              for (size_t j = 0; j < n1->a.size() && j < 65536; ++j)
                if (P wv = f_.resolve(n1->a[j])) f.w[(uint32_t)c0->n + (uint32_t)j] = wv->n;
              k += 2;
            } else if (k + 2 < W->a.size()) {
              P wv = f_.resolve(W->a[k + 2]);
              for (uint32_t c = (uint32_t)c0->n; c <= (uint32_t)n1->n && c - (uint32_t)c0->n < 65536; ++c) f.w[c] = wv ? wv->n : f.dw;
              k += 3;
            } else break;
          }
      }
    } else {
      P fc = f_.get(fo, "FirstChar"), W = f_.get(fo, "Widths");
      if (fc && W && W->t == Obj::Arr)
        for (size_t j = 0; j < W->a.size() && j < 256; ++j)
          if (P wv = f_.resolve(W->a[j])) f.w[(uint32_t)fc->n + (uint32_t)j] = wv->n;
      // (no widths: one of the 14 standard fonts. A narrow guess, so the next word's start still shows its gap:
      //  a space too many costs less than words run together)
      if (f.w.empty()) f.dw = 300;
    }
    if (P bf = f_.get(fo, "BaseFont"); bf) {
      std::string n = bf->s;
      for (char& ch : n) ch = (char)tolower((unsigned char)ch);
      f.bold = n.find("bold") != std::string::npos || n.find("black") != std::string::npos || n.find("heavy") != std::string::npos;
    }
    if (P enc = f_.get(fo, "Encoding"); enc && enc->t == Obj::Dict)
      if (P diff = f_.get(enc, "Differences"); diff && diff->t == Obj::Arr) {
        uint32_t code = 0;
        for (auto& x : diff->a) {
          P v = f_.resolve(x);
          if (!v) continue;
          if (v->t == Obj::Num) code = (uint32_t)v->n;
          else if (v->t == Obj::Name) {
            if (const uint32_t u = glyph_unicode(v->s)) { std::string s; put_utf8(s, u); f.map[code] = s; }
            ++code;
          }
        }
      }
    if (P tu = f_.get(fo, "ToUnicode"); tu && tu->has_stream) {
      const int bytes = f.bytes;
      parse_cmap(f_.decode(tu), f);
      if (f.map.empty()) f.bytes = bytes;
    }
    return fonts_[fo.get()] = f;
  }
  void run(const std::string& cs, const P& res, int depth, M ctm) {
    if (depth > 4) return;
    Lexer lx{cs, 0};
    std::vector<P> st;
    std::vector<M> stack;
    M tm{1, 0, 0, 1, 0, 0}, tlm = tm;
    double size = 0, lead = 0, tc = 0, tw = 0, th = 1;
    const Font* fnt = nullptr;
    P fonts = f_.get(res, "Font"), xobj = f_.get(res, "XObject");
    auto num = [&](size_t k) { return k < st.size() && st[k] && st[k]->t == Obj::Num ? st[k]->n : 0.0; };
    auto show = [&](const std::string& codes) {
      if (!fnt) return;
      const M r = mul(tm, ctm);
      const double y = r[5], sz = size * std::sqrt(std::abs(r[2] * r[2] + r[3] * r[3]));
      if (std::abs(y - cur_y_) > std::max(1.0, sz * 0.5)) {  // (another baseline: a new line)
        flush();
        cur_y_ = y;
        cur_size_ = sz;
      } else if (!fnt->w.empty() && r[4] > last_x_ + std::max(1.0, sz * 1.8) && !cur_.empty()) {  // (a column gap; known widths only)
        cur_ = tidy(cur_) + " | ";
        columns_ = true;
      } else if (r[4] > last_x_ + std::max(0.5, sz * 0.15) && !cur_.empty() && cur_.back() != ' ') cur_ += ' ';
      cur_size_ = std::max(cur_size_, sz);
      const std::string t = fnt->text(codes);
      cur_ += t;
      const size_t visible = (size_t)std::count_if(t.begin(), t.end(), [](char ch) { return ch != ' '; });
      chars_ += visible;
      if (fnt->bold) bold_chars_ += visible;
      tm = mul({1, 0, 0, 1, fnt->advance(codes, size, tc, tw) * th, 0}, tm);
      last_x_ = mul(tm, ctm)[4];
    };
    while (lx.i < cs.size()) {
      if (++steps_ > 20000000) return;
      std::string op;
      P v = lx.value(&op);
      if (v) {
        st.push_back(v);
        if (st.size() > 4096) st.clear();
        continue;
      }
      if (op.empty()) {
        if (lx.i >= cs.size()) break;
        continue;
      }
      if (op == "BI") {  // (an inline image: its data up to EI)
        const size_t e = cs.find("EI", lx.i);
        lx.i = e == std::string::npos ? cs.size() : e + 2;
      } else if (op == "q") stack.push_back(ctm);
      else if (op == "Q") {
        if (!stack.empty()) { ctm = stack.back(); stack.pop_back(); }
      } else if (op == "cm" && st.size() >= 6) ctm = mul({num(0), num(1), num(2), num(3), num(4), num(5)}, ctm);
      else if (op == "BT") tm = tlm = {1, 0, 0, 1, 0, 0};
      else if (op == "Tf" && st.size() >= 2) {
        size = num(1);
        fnt = nullptr;
        if (st[0] && st[0]->t == Obj::Name)
          if (P fo = f_.get(fonts, st[0]->s)) fnt = &font(fo);
      } else if (op == "TL") lead = num(0);
      else if (op == "Tc") tc = num(0);
      else if (op == "Tw") tw = num(0);
      else if (op == "Tz") th = num(0) / 100.0;
      else if (op == "Td" || op == "TD") {
        if (op == "TD") lead = -num(1);
        tlm = mul({1, 0, 0, 1, num(0), num(1)}, tlm);
        tm = tlm;
      } else if (op == "Tm" && st.size() >= 6) tm = tlm = {num(0), num(1), num(2), num(3), num(4), num(5)};
      else if (op == "T*") { tlm = mul({1, 0, 0, 1, 0, -lead}, tlm); tm = tlm; }
      else if (op == "Tj" && !st.empty() && st.back()->t == Obj::Str) show(st.back()->s);
      else if ((op == "'" || op == "\"") && !st.empty() && st.back()->t == Obj::Str) {
        if (op == "\"" && st.size() >= 3) { tw = num(0); tc = num(1); }
        tlm = mul({1, 0, 0, 1, 0, -lead}, tlm);
        tm = tlm;
        show(st.back()->s);
      } else if (op == "TJ" && !st.empty() && st.back()->t == Obj::Arr) {
        for (auto& x : st.back()->a) {
          if (x->t == Obj::Str) show(x->s);
          else if (x->t == Obj::Num) tm = mul({1, 0, 0, 1, -x->n / 1000.0 * size * th, 0}, tm);
        }
      } else if (op == "Do" && !st.empty() && st[0]->t == Obj::Name) {
        P xo = f_.get(xobj, st[0]->s);
        if (P sub = f_.get(xo, "Subtype"); sub && sub->s == "Form") {
          M m = ctm;
          if (P mx = f_.get(xo, "Matrix"); mx && mx->t == Obj::Arr && mx->a.size() == 6) {
            M fm;
            for (int k = 0; k < 6; ++k) fm[k] = mx->a[k]->n;
            m = mul(fm, ctm);
          }
          P r2 = f_.get(xo, "Resources");
          run(f_.decode(xo), r2 ? r2 : res, depth + 1, m);
        }
      }
      st.clear();
    }
  }
};

}  // namespace pdf

inline Doc read_pdf(const std::string& data) {
  Doc d;
  if (data.compare(0, 5, "%PDF-") != 0 && data.find("%PDF-") > 1024) {
    d.note = "not a PDF";
    return d;
  }
  pdf::File f(data);
  if (f.encrypted) {
    d.note = "an encrypted PDF: its text cannot be read";
    return d;
  }
  pdf::Reader r(f);
  const auto pages = f.pages();
  for (size_t k = 0; k < pages.size(); ++k) r.page(pages[k].first, pages[k].second, (int)k + 1);
  d.pages = (int)pages.size();
  if (d.pages >= 2) {
    auto key = [](const pdf::Line& l) {  // (the text without its digits, and its height on the page)
      std::string t;
      for (char c : l.text)
        if (!isdigit((unsigned char)c)) t += c;
      return t + "@" + std::to_string((int)std::lround(l.y / 4));
    };
    std::map<std::string, std::set<int>> on;
    for (const auto& l : r.lines) on[key(l)].insert(l.page);
    std::vector<pdf::Line> kept;
    for (const auto& l : r.lines)
      if (on[key(l)].size() * 2 < (size_t)d.pages + (d.pages == 2 ? 1 : 0) || l.text.size() > 150) kept.push_back(l);
    r.lines.swap(kept);
  }
  // the body text's size: the size with the most characters; larger short paragraphs are headings
  std::map<int, size_t> by_size;
  for (const auto& l : r.lines) by_size[(int)std::lround(l.size * 2)] += l.text.size();
  int body = 0;
  size_t most = 0;
  for (const auto& [s, n] : by_size)
    if (n > most) { most = n; body = s; }
  const double body_size = body / 2.0;
  std::vector<double> head_sizes;
  for (const auto& [s, n] : by_size)
    if (s / 2.0 >= body_size * 1.15) head_sizes.push_back(s / 2.0);
  std::sort(head_sizes.rbegin(), head_sizes.rend());
  if (head_sizes.size() > 5) head_sizes.resize(5);
  auto level_of = [&](double sz) {
    for (size_t k = 0; k < head_sizes.size(); ++k)
      if (std::abs(sz - head_sizes[k]) < 0.3) return (int)k + 1;
    return 0;
  };
  // lines into paragraphs: a new one at another page, another size or style, or a gap clearly larger than a line's
  Para cur;
  double prev_y = 0, prev_size = 0, gap = 0;
  bool prev_bold = false, all_bold = true, prev_columns = false;
  const int bold_level = std::min(6, (int)head_sizes.size() + 1);
  auto end_para = [&] {
    cur.text = tidy(cur.text);
    if (!cur.text.empty()) {
      if (cur.level && cur.text.size() > 200) cur.level = 0;
      // (a short bold paragraph that is not a sentence, e.g. "BAB II LANDASAN TEORI": a heading)
      if (!cur.level && all_bold && cur.text.size() < 120 && cur.text.back() != '.' && cur.text.back() != ':' &&
          cur.text.back() != ',')
        cur.level = bold_level;
      d.paras.push_back(cur);
    }
    cur = Para();
    all_bold = true;
  };
  for (const auto& l : r.lines) {
    const int lvl = level_of(l.size);
    const double dy = prev_y - l.y;
    auto item = [](const std::string& t) {  // (a list item's marker: "•", "-", "o ", "1.", "a)")
      if (t.rfind("\xe2\x80\xa2", 0) == 0 || t.rfind("\xe2\x80\x93 ", 0) == 0 || t.rfind("- ", 0) == 0 || t.rfind("o ", 0) == 0 ||
          t.rfind("\xef\x82\xb7", 0) == 0)
        return true;
      size_t k = 0;
      while (k < t.size() && k < 3 && isalnum((unsigned char)t[k])) ++k;
      return k > 0 && k < t.size() && (t[k] == '.' || t[k] == ')') && k + 1 < t.size() && t[k + 1] == ' ' &&
             (isdigit((unsigned char)t[0]) || k == 1);
    };
    const bool same = !cur.text.empty() && !l.columns && !prev_columns && !item(l.text) && l.page == cur.page && std::abs(l.size - prev_size) < 0.6 && lvl == cur.level &&
                      dy > 0 && dy < std::max(l.size, prev_size) * 1.9 && (gap == 0 || dy < gap * 1.35) && l.bold == prev_bold;
    if (!same) {
      end_para();
      cur.page = l.page;
      cur.level = l.columns ? 0 : lvl;
      gap = 0;
    } else if (gap == 0) gap = dy;
    if (!cur.text.empty() && cur.text.back() == '-' && cur.text.size() > 1 && isalpha((unsigned char)cur.text[cur.text.size() - 2]))
      cur.text.pop_back();  // (a word broken at the line's end)
    else if (!cur.text.empty()) cur.text += ' ';
    cur.text += l.text;
    all_bold = all_bold && l.bold && !l.columns;
    prev_columns = l.columns;
    prev_y = l.y;
    prev_size = l.size;
    prev_bold = l.bold;
  }
  end_para();
  if (d.paras.empty()) d.note = d.pages ? "the PDF has no text layer (a scan or images: it needs OCR)" : "no pages found in the PDF";
  else if (d.pages >= 3 && d.chars() < (size_t)d.pages * 80) {  // (a few words on most pages: scanned pages)
    int with_text = 0;
    std::set<int> seen;
    for (const Para& q : d.paras) seen.insert(q.page);
    with_text = (int)seen.size();
    d.note = "little text: " + std::to_string(d.chars()) + " characters on " + std::to_string(d.pages) + " pages (" +
             std::to_string(with_text) + " with any); the rest are probably scanned images, which need OCR";
  }
  return d;
}

// text and Markdown: paragraphs at blank lines, "#" headings
inline Doc read_text(const std::string& data, bool markdown) {
  Doc d;
  std::string t = data;
  if (t.size() >= 3 && (unsigned char)t[0] == 0xEF && (unsigned char)t[1] == 0xBB && (unsigned char)t[2] == 0xBF) t.erase(0, 3);
  if (!valid_utf8(t)) {  // (an ANSI file: Windows-1252)
    std::string u;
    for (unsigned char c : t) put_utf8(u, cp1252(c));
    t.swap(u);
  }
  std::string cur;
  auto end_para = [&](int level) {
    std::string s = cur;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    if (!s.empty()) d.paras.push_back({s, 1, level});
    cur.clear();
  };
  size_t i = 0;
  while (i <= t.size()) {
    const size_t e = std::min(t.find('\n', i), t.size());
    std::string line = t.substr(i, e - i);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    i = e + 1;
    if (markdown && !line.empty() && line[0] == '#') {
      size_t h = 0;
      while (h < line.size() && line[h] == '#') ++h;
      if (h <= 6 && h < line.size() && line[h] == ' ') {
        end_para(0);
        cur = tidy(line.substr(h + 1));
        end_para((int)h);
        continue;
      }
    }
    if (tidy(line).empty()) end_para(0);
    else cur += line + "\n";
    if (e >= t.size()) break;
  }
  end_para(0);
  if (d.paras.empty()) d.note = "the file is empty";
  return d;
}

inline std::string lower_ext(const std::string& name) {
  const size_t p = name.rfind('.');
  std::string e = p == std::string::npos ? "" : name.substr(p + 1);
  for (char& c : e) c = (char)tolower((unsigned char)c);
  return e;
}
inline Doc read_xlsx(const std::string& zip);            // (tools/xlsx.h, included at the end)
inline Doc read_csv(const std::string& data, char delim);
// a file's text by its name and bytes; binary formats it cannot read: a note
inline Doc read(const std::string& name, const std::string& data) {
  const std::string e = lower_ext(name);
  if (data.empty()) {
    Doc d;
    d.note = "the file is empty";
    return d;
  }
  if (e == "docx" || e == "docm" || e == "dotx") return read_docx(data);
  if (e == "xlsx" || e == "xlsm" || e == "xls") return read_xlsx(data);
  if (e == "pdf" || data.compare(0, 5, "%PDF-") == 0) return read_pdf(data);
  if (data.find('\0') < std::min<size_t>(data.size(), 8192)) {
    Doc d;
    d.note = e == "doc" ? "a Word 97-2003 .doc (binary): save it as DOCX to read it" : "a binary file (" + e + "): its text cannot be read";
    return d;
  }
  if (e == "csv" || e == "tsv") return read_csv(data, e == "tsv" ? '\t' : 0);
  return read_text(data, e == "md" || e == "markdown");
}

// ---------------------------------------------------------------- outline, chunks, BM25
struct Section {
  std::string title;
  int level = 1, page = 1;
  size_t first = 0, last = 0;  // (paragraphs [first, last): up to the next heading)
  size_t chars = 0;
  size_t span = 0, span_chars = 0;  // (with its subsections: up to the next heading of its level or higher)
  int page_to = 1;
};
struct Chunk {
  size_t first, last;  // (paragraphs)
  int page_from, page_to, section;
  std::string text;
};

// words for the index: lower case, Indonesian particles (-nya -lah -kah -pun) taken off, common words left out
inline std::vector<std::string> words(const std::string& s) {
  static const std::set<std::string> stop = {"yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", "atau", "pada",
                                             "dalam", "adalah", "akan", "juga", "tidak", "sebagai", "oleh", "the", "of", "and",
                                             "to", "in", "a", "an", "is", "are", "for", "on", "with", "as", "by", "be", "or", "it"};
  std::vector<std::string> v;
  std::string w;
  auto end = [&] {
    if (w.size() > 6)
      for (const char* suf : {"nya", "lah", "kah", "pun"})
        if (w.size() > 6 && w.compare(w.size() - 3, 3, suf) == 0) { w.resize(w.size() - 3); break; }
    if (!w.empty() && !stop.count(w)) v.push_back(w);
    w.clear();
  };
  for (size_t i = 0; i < s.size(); ++i) {
    const unsigned char c = (unsigned char)s[i];
    if (c >= 0x80) {  // (a letter of another script: part of the word, whole)
      w += (char)c;
    } else if (isalnum(c)) w += (char)tolower(c);
    else end();
  }
  end();
  return v;
}

class Index {
 public:
  Doc doc;
  std::vector<Section> outline;  // (headings; for a document without them: its pages, or parts)
  std::vector<Chunk> chunks;
  bool by_headings = false;

  // the deepest heading level that keeps the outline at max entries or fewer (at least the top level)
  int outline_depth(size_t max) const {
    int depth = 1;
    for (int l = 1; l <= 6; ++l) {
      size_t n = 0;
      for (const Section& s : outline) n += s.level <= l;
      if (n > max) break;
      depth = l;
    }
    // (only a title or two on top, e.g. the document's title over its chapters: the next level too, cut at 2 x max)
    size_t top = 0;
    for (const Section& s : outline) top += s.level <= depth;
    return top < 8 && depth < 6 ? depth + 1 : depth;
  }

  explicit Index(Doc d, size_t chunk_chars = 2400) : doc(std::move(d)) {
    // sections: from the headings
    for (size_t k = 0; k < doc.paras.size(); ++k)
      if (doc.paras[k].level) {
        if (!outline.empty()) outline.back().last = k;
        outline.push_back({doc.paras[k].text, doc.paras[k].level, doc.paras[k].page, k, doc.paras.size(), 0});
      }
    by_headings = outline.size() >= 2;
    if (!by_headings) {  // (pages, or parts of ~chunk_chars * 4 for a text without pages)
      outline.clear();
      size_t start = 0, acc = 0;
      for (size_t k = 0; k <= doc.paras.size(); ++k) {
        const bool cut = k == doc.paras.size() ||
                         (doc.pages > 1 ? doc.paras[k].page != doc.paras[start].page : acc > chunk_chars * 4);
        if (cut && k > start) {
          Section s;
          s.first = start;
          s.last = k;
          s.page = doc.paras[start].page;
          s.level = 1;
          const std::string& t = doc.paras[start].text;
          s.title = (doc.pages > 1 ? "page " + std::to_string(s.page) : "part " + std::to_string(outline.size() + 1)) + ": " + cut_words(t, 60);
          outline.push_back(s);
          start = k;
          acc = 0;
        }
        if (k < doc.paras.size()) acc += doc.paras[k].text.size();
      }
    } else if (outline[0].first > 0)  // (text before the first heading)
      outline.insert(outline.begin(), {"(before the first heading)", 1, doc.paras[0].page, 0, outline[0].first, 0});
    for (size_t si = 0; si < outline.size(); ++si) {
      Section& s = outline[si];
      for (size_t k = s.first; k < s.last; ++k) s.chars += doc.paras[k].text.size() + 2;
      s.span = doc.paras.size();
      if (!by_headings) s.span = s.last;
      else
        for (size_t j = si + 1; j < outline.size(); ++j)
          if (outline[j].level <= s.level) { s.span = outline[j].first; break; }
      for (size_t k = s.first; k < s.span; ++k) s.span_chars += doc.paras[k].text.size() + 2;
      s.page_to = s.span > s.first ? doc.paras[s.span - 1].page : s.page;
    }
    // chunks: paragraphs together up to chunk_chars, never across a section the outline shows (a deeper heading, e.g.
    // a slide's or a table's bold line, stays inside its chunk: a chunk of one line finds nothing); a longer
    // paragraph cut at sentences
    std::vector<size_t> majors;
    const int depth = outline_depth(60);
    for (size_t si = 0; si < outline.size(); ++si)
      if (!by_headings || outline[si].level <= depth || si == 0) majors.push_back(si);
    for (size_t mi = 0; mi < majors.size(); ++mi) {
      const size_t si = majors[mi];
      const Section& s = outline[si];
      const size_t end = mi + 1 < majors.size() ? outline[majors[mi + 1]].first : (by_headings ? doc.paras.size() : s.last);
      Chunk c{s.first, s.first, doc.paras[s.first].page, doc.paras[s.first].page, (int)si, ""};
      auto emit = [&] {
        if (!c.text.empty()) chunks.push_back(c);
        c.text.clear();
      };
      for (size_t k = s.first; k < end; ++k) {
        const Para& p = doc.paras[k];
        std::string t = p.level ? std::string((size_t)p.level, '#') + " " + p.text : p.text;
        if (!c.text.empty() && c.text.size() + t.size() > chunk_chars) emit();
        while (t.size() > chunk_chars * 3 / 2) {  // (a very long paragraph: pieces at a sentence end)
          size_t cut = t.rfind(". ", chunk_chars);
          if (cut == std::string::npos || cut < chunk_chars / 2) cut = t.rfind(' ', chunk_chars);
          if (cut == std::string::npos || cut == 0) cut = chunk_chars;
          if (c.text.empty()) { c.first = k; c.page_from = p.page; }
          c.text = t.substr(0, cut + 1);
          c.last = k + 1;
          c.page_to = p.page;
          emit();
          t.erase(0, cut + 1);
        }
        if (c.text.empty()) { c.first = k; c.page_from = p.page; }
        c.text += (c.text.empty() ? "" : "\n\n") + t;
        c.last = k + 1;
        c.page_to = p.page;
      }
      emit();
    }
    // BM25 statistics
    df_.clear();
    tf_.resize(chunks.size());
    len_.resize(chunks.size());
    double total = 0;
    for (size_t k = 0; k < chunks.size(); ++k) {
      for (const std::string& w : words(chunks[k].text)) tf_[k][w]++;
      len_[k] = 0;
      for (const auto& [w, n] : tf_[k]) { df_[w]++; len_[k] += n; }
      total += len_[k];
    }
    avg_ = chunks.empty() ? 1 : std::max(1.0, total / chunks.size());
  }

  static std::string cut_words(const std::string& s, size_t n) {
    if (s.size() <= n) return s;
    size_t c = s.rfind(' ', n);
    if (c == std::string::npos || c < n / 2) c = n;
    while (c > 0 && ((unsigned char)s[c] & 0xC0) == 0x80) --c;  // (not inside a character)
    return s.substr(0, c) + "...";
  }
  // the text of paragraphs [first, last), with "[page N]" where a page starts
  std::string text(size_t first, size_t last) const {
    std::string o;
    int page = 0;
    for (size_t k = first; k < last && k < doc.paras.size(); ++k) {
      const Para& p = doc.paras[k];
      if (doc.pages > 1 && p.page != page) {
        o += (o.empty() ? "" : "\n\n") + std::string("[page ") + std::to_string(p.page) + "]";
        page = p.page;
      }
      o += (o.empty() ? "" : "\n\n") + (p.level ? std::string((size_t)p.level, '#') + " " : "") + p.text;
    }
    return o;
  }
  struct Hit {
    size_t chunk;
    double score;
  };
  // BM25 (k1 1.2, b 0.75); a query word of 5+ letters also finds longer words that start with it (weight 0.6)
  std::vector<Hit> search(const std::string& query, size_t top) const {
    std::vector<double> score(chunks.size(), 0);
    const double N = (double)chunks.size();
    std::set<std::string> qs;
    for (const std::string& w : words(query)) qs.insert(w);
    for (const std::string& q : qs) {
      std::vector<std::pair<std::string, double>> terms;
      if (df_.count(q)) terms.push_back({q, 1.0});
      if (q.size() >= 5)
        for (auto it = df_.lower_bound(q); it != df_.end() && it->first.compare(0, q.size(), q) == 0; ++it)
          if (it->first != q) terms.push_back({it->first, 0.6});
      for (const auto& [t, wgt] : terms) {
        const double n = df_.at(t), idf = std::log(1 + (N - n + 0.5) / (n + 0.5));
        for (size_t k = 0; k < chunks.size(); ++k) {
          auto f = tf_[k].find(t);
          if (f == tf_[k].end()) continue;
          const double tf = f->second;
          score[k] += wgt * idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len_[k] / avg_));
        }
      }
    }
    std::vector<Hit> v;
    for (size_t k = 0; k < chunks.size(); ++k)
      if (score[k] > 0) v.push_back({k, score[k]});
    std::sort(v.begin(), v.end(), [](const Hit& a, const Hit& b) { return a.score > b.score; });
    if (v.size() > top) v.resize(top);
    return v;
  }

 private:
  std::map<std::string, int> df_;
  std::vector<std::unordered_map<std::string, int>> tf_;
  std::vector<double> len_;
  double avg_ = 1;
};

}  // namespace rdoc

#include "xlsx.h"  // (read_xlsx, read_csv: they use Doc)
