// Excel workbooks with no library: uploaded .xlsx / .xlsm (and .csv / .tsv) read as an rdoc::Doc for the web UI's
// file tools (tools/doc_reader.h calls read_xlsx / read_csv by the file's extension), and create_file's "xlsx"
// written from the Markdown the model already writes (write()).
//
// Reading: xl/workbook.xml (the sheets in their order, with names; xl/_rels/workbook.xml.rels for each sheet's part),
// xl/sharedStrings.xml (rich text runs joined, phonetic runs left out), xl/styles.xml (each cell style's number
// format: dates and times, percentages, a currency prefix such as "Rp") and each sheet's cells: shared and inline
// strings, numbers, booleans, errors, formula cells as their cached value (the formula itself is not shown). Dates
// come out as yyyy-mm-dd (plus hh:mm when there is a time); a merged range keeps its value in its first cell only.
// Each sheet is a heading "Sheet: <name>" and its rows are paragraphs "A | B | C"; when the first row looks like a
// header (text in most of the used columns) every row repeats the column names, "Nama: Budi | Nilai: 80", so a
// passage search_file finds still says what each value is. Empty rows and columns are left out; at most 200,000
// cells are read (a note says so). CSV: the same rows (delimiter , ; or tab, whichever the first line has most).
//
// Writing: each Markdown table becomes a sheet, named by the last heading before it (since the previous table; at
// most 31 characters, without [ ] : * ? / \), else "Sheet1", "Sheet2" ... Text outside the tables (paragraphs,
// lists, code, headings) goes to a first sheet "Notes", one block per row, with a line where each table was ("table:
// sheet 'X'"); the workbook still opens on the first table. With no text outside the tables there is no Notes sheet.
// Cells: numbers as numbers with a number format ("1.234,5", "1,234.5", "Rp 1.234.000", "$12.50", "15 %"). Indonesian
// or US style: "1.500" and "1,500" alone cannot tell (1500, or 1.5?), so each table goes by its numbers that do
// ("1.234,5", "1.234.567", "2,5" Indonesian; "1,234.5", "1,234,567", "3.14" US), else the whole document's, else
// Indonesian; write(..., "id" / "us") fixes it. With a currency a separator before three digits is always thousands
// (Excel then shows the numbers in the computer's own style either way). Dates ("2026-10-27", "27/10/2026", "27-10-2026", with an optional
// hh:mm) as real dates; numbers with a leading zero ("0812...") or over 15 digits stay text, the rest are shared
// strings (a text starting with "=" stays text, never a formula). The header row is bold on a fill, frozen under
// it, with an autofilter; column widths come from the content. The parts are the minimal set Excel opens without
// repair, each in the schema's element order, in a stored ZIP (doc::zip).
#pragma once
#include <cmath>
#include "doc_reader.h"
#include "doc_writer.h"

namespace xlsx {

// ---------------------------------------------------------------- XML
inline bool self_closing(const std::string& tag) { return tag.size() >= 2 && tag[tag.size() - 2] == '/'; }
// the tags of x in order: f(name without its namespace prefix ("/row" for an end tag), the whole tag, the text after
// it up to the next tag as [a, b)); f returns false to stop. A '>' inside a quoted attribute does not end a tag; no
// recursion, so deep nesting costs nothing
template <class F>
inline void walk(const std::string& x, F&& f) {
  size_t i = x.find('<');
  while (i != std::string::npos) {
    if (x.compare(i, 4, "<!--") == 0) {
      const size_t c = x.find("-->", i);
      i = c == std::string::npos ? c : x.find('<', c);
      continue;
    }
    size_t e = i + 1;
    for (char q = 0; e < x.size(); ++e) {
      const char c = x[e];
      if (q) { if (c == q) q = 0; }
      else if (c == '"' || c == '\'') q = c;
      else if (c == '>') break;
    }
    if (e >= x.size()) return;
    const bool end = x[i + 1] == '/';
    size_t a = i + 1 + end, ne = a;
    size_t colon = 0;
    while (ne < e && x[ne] != ' ' && x[ne] != '/' && x[ne] != '\t' && x[ne] != '\r' && x[ne] != '\n') {
      if (x[ne] == ':') colon = ne;
      ++ne;
    }
    if (colon) a = colon + 1;
    const size_t next = x.find('<', e);
    if (!f((end ? "/" : "") + x.substr(a, ne - a), x.substr(i, e - i + 1), e + 1, next == std::string::npos ? x.size() : next)) return;
    i = next;
  }
}
// an attribute's value (either quote), entities decoded; "" when missing
inline std::string attr(const std::string& tag, const char* name) {
  const size_t n = strlen(name);
  for (size_t p = tag.find(name); p != std::string::npos; p = tag.find(name, p + 1)) {
    if (p == 0 || !isspace((unsigned char)tag[p - 1])) continue;
    size_t k = p + n;
    while (k < tag.size() && isspace((unsigned char)tag[k])) ++k;
    if (k >= tag.size() || tag[k] != '=') continue;
    for (++k; k < tag.size() && isspace((unsigned char)tag[k]); ++k) {}
    if (k >= tag.size() || (tag[k] != '"' && tag[k] != '\'')) continue;
    const size_t e = tag.find(tag[k], k + 1);
    return e == std::string::npos ? "" : rdoc::xml_text(tag.substr(k + 1, e - k - 1));
  }
  return "";
}
// "B12" -> column 2, row 12 (1-based; "$" ignored); false when not a cell reference within Excel's limits
inline bool cell_ref(const std::string& r, int& col, int& row) {
  size_t i = 0;
  int c = 0, w = 0;
  if (i < r.size() && r[i] == '$') ++i;
  for (; i < r.size() && isalpha((unsigned char)r[i]) && c <= 16384; ++i) c = c * 26 + (toupper((unsigned char)r[i]) - 'A' + 1);
  if (i < r.size() && r[i] == '$') ++i;
  const size_t d = i;
  for (; i < r.size() && isdigit((unsigned char)r[i]) && w <= 1048576; ++i) w = w * 10 + (r[i] - '0');
  if (c < 1 || c > 16384 || i == d || i != r.size() || w < 1 || w > 1048576) return false;
  col = c;
  row = w;
  return true;
}

// ---------------------------------------------------------------- dates
inline long long days_from_civil(long long y, int m, int d) {  // (days since 1970-01-01)
  y -= m <= 2;
  const long long era = (y >= 0 ? y : y - 399) / 400, yoe = y - era * 400;
  const long long doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1, doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + doe - 719468;
}
inline std::string civil(long long z) {  // (days since 1970-01-01 -> "yyyy-mm-dd")
  z += 719468;
  const long long era = (z >= 0 ? z : z - 146096) / 146097, doe = z - era * 146097;
  const long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365, doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const long long mp = (5 * doy + 2) / 153, d = doy - (153 * mp + 2) / 5 + 1, m = mp < 10 ? mp + 3 : mp - 9;
  char b[32];
  snprintf(b, sizeof b, "%04lld-%02lld-%02lld", yoe + era * 400 + (m <= 2), m, d);
  return b;
}
// an Excel serial date (days since 1899-12-30, with Excel's 1900-02-29) as text: the date, its time when it has one
inline std::string serial_text(double s, bool time_only, bool d1904) {
  if (d1904) s += 1462;
  if (!(s >= 0 && s < 2958466)) {
    char b[40];
    snprintf(b, sizeof b, "%.15g", s);
    return b;
  }
  long long day = (long long)std::floor(s), secs = std::llround((s - (double)day) * 86400);
  if (secs >= 86400) { ++day; secs -= 86400; }
  std::string o;
  if (!time_only || day >= 1) o = day == 60 ? "1900-02-29" : civil(day < 60 ? day - 25568 : day - 25569);
  if (secs || time_only) {
    char b[24];
    if (secs % 60) snprintf(b, sizeof b, "%02lld:%02lld:%02lld", secs / 3600, secs / 60 % 60, secs % 60);
    else snprintf(b, sizeof b, "%02lld:%02lld", secs / 3600, secs / 60 % 60);
    o += (o.empty() ? "" : " ") + std::string(b);
  }
  return o;
}

// ---------------------------------------------------------------- reading
struct Fmt {
  enum Kind { Number, Date, Time, Elapsed, Percent } kind = Number;
  std::string prefix, suffix;  // (a number's literal text, e.g. "Rp ")
};
// a number format by its id (built-in) or code (custom): dates have d / m / y, times h / s (or m with them), outside
// quotes, escapes and [..] (except [h] [mm] [ss], elapsed time)
inline Fmt fmt_of(int id, const std::string& code) {
  Fmt f;
  if (code.empty()) {
    if (id == 9 || id == 10) f.kind = Fmt::Percent;
    else if ((id >= 14 && id <= 17) || id == 22 || (id >= 27 && id <= 31) || id == 36 || (id >= 50 && id <= 58)) f.kind = Fmt::Date;
    else if (id == 46) f.kind = Fmt::Elapsed;  // ([h]:mm:ss)
    else if ((id >= 18 && id <= 21) || (id >= 32 && id <= 35) || id == 45 || id == 47) f.kind = Fmt::Time;
    return f;
  }
  bool d = false, t = false, m = false, pct = false, digit = false, el = false;
  for (size_t i = 0; i < code.size(); ++i) {
    const char c = code[i];
    std::string& lit = digit ? f.suffix : f.prefix;
    if (c == ';') break;
    if (c == '"') {
      size_t e = code.find('"', i + 1);
      if (e == std::string::npos) e = code.size();
      lit += code.substr(i + 1, e - i - 1);
      i = e;
    } else if (c == '\\') {
      if (i + 1 < code.size()) lit += code[++i];
    } else if (c == '_' || c == '*') ++i;
    else if (c == '[') {
      const size_t e = code.find(']', i);
      if (e == std::string::npos) break;
      const std::string in = code.substr(i + 1, e - i - 1);
      if (!in.empty() && in[0] == '$') lit += in.substr(1, in.find('-') == std::string::npos ? std::string::npos : in.find('-') - 1);
      else if (!in.empty() && in.find_first_not_of("hHmMsS") == std::string::npos) t = el = true;
      i = e;
    } else {
      const char l = (char)tolower((unsigned char)c);
      if (l == 'd' || l == 'y') d = true;
      else if (l == 'h' || l == 's') t = true;
      else if (l == 'm') m = true;
      else if (c == '%') pct = true;
      else if (c == '0' || c == '#' || c == '?') digit = true;
      else if (c == '$' || c == ' ') lit += c;
    }
  }
  if (d || (m && !t)) f.kind = Fmt::Date;
  else if (el) f.kind = Fmt::Elapsed;  // ([h]:mm: a duration, 30:00 for 1.25 days)
  else if (t || m) f.kind = Fmt::Time;
  else if (pct) f.kind = Fmt::Percent;
  if (f.kind != Fmt::Number) f.prefix.clear(), f.suffix.clear();
  return f;
}
inline std::string value_text(double v, const Fmt& f, bool d1904) {
  char b[48];
  if (f.kind == Fmt::Date || f.kind == Fmt::Time) return serial_text(v, f.kind == Fmt::Time, d1904);
  if (f.kind == Fmt::Elapsed) {
    const long long secs = std::llround(std::fabs(v) * 86400);
    if (secs % 60) snprintf(b, sizeof b, "%s%lld:%02lld:%02lld", v < 0 ? "-" : "", secs / 3600, secs / 60 % 60, secs % 60);
    else snprintf(b, sizeof b, "%s%lld:%02lld", v < 0 ? "-" : "", secs / 3600, secs / 60 % 60);
    return b;
  }
  if (f.kind == Fmt::Percent) {
    snprintf(b, sizeof b, "%.10g%%", v * 100);
    return b;
  }
  snprintf(b, sizeof b, "%.15g", v);
  return rdoc::tidy(f.prefix + b + f.suffix);
}

struct Cell {
  std::string text;
  bool str = false;  // (a string cell, not a number, date or boolean)
};
struct Grid {
  std::string name;
  std::map<int, std::map<int, Cell>> rows;  // (row -> column -> cell, 1-based)
  size_t cells = 0;
  bool cut = false;  // (stopped at the cell budget)
};

inline std::vector<std::string> shared_strings(const std::string& x) {
  std::vector<std::string> v;
  std::string cur;
  bool in_si = false, rph = false;
  walk(x, [&](const std::string& n, const std::string& tag, size_t a, size_t b) {
    if (n == "si") {
      cur.clear();
      in_si = !self_closing(tag);
      if (!in_si) v.push_back("");
    } else if (n == "/si" && in_si) {
      v.push_back(cur);
      in_si = false;
    } else if (n == "rPh") rph = !self_closing(tag);
    else if (n == "/rPh") rph = false;
    else if (n == "t" && in_si && !rph && !self_closing(tag)) cur += rdoc::xml_text(x.substr(a, b - a));
    return v.size() < 4000000;
  });
  return v;
}

// a sheet's cells into g (at most budget cells)
inline void read_sheet(const std::string& x, const std::vector<std::string>& sst, const std::vector<Fmt>& xfs, bool d1904,
                       Grid& g, size_t budget) {
  int row = 0, col = 0, style = 0;
  std::string type, v, is;
  bool in_c = false, in_is = false, rph = false;
  std::vector<std::array<int, 4>> merges;  // (c1, r1, c2, r2)
  walk(x, [&](const std::string& n, const std::string& tag, size_t a, size_t b) {
    if (n == "row") {
      const int r = atoi(attr(tag, "r").c_str());
      row = r > 0 ? r : row + 1;
      col = 0;
    } else if (n == "c") {
      int c = 0, r = 0;
      if (cell_ref(attr(tag, "r"), c, r)) { col = c; row = r; }
      else ++col;
      if (row < 1) row = 1;
      type = attr(tag, "t");
      style = atoi(attr(tag, "s").c_str());
      v.clear();
      is.clear();
      in_c = !self_closing(tag);
    } else if (n == "v" && in_c && !self_closing(tag)) v += rdoc::xml_text(x.substr(a, b - a));
    else if (n == "is") in_is = !self_closing(tag);
    else if (n == "/is") in_is = false;
    else if (n == "rPh") rph = !self_closing(tag);
    else if (n == "/rPh") rph = false;
    else if (n == "t" && in_c && in_is && !rph && !self_closing(tag)) is += rdoc::xml_text(x.substr(a, b - a));
    else if (n == "/c" && in_c) {
      in_c = false;
      std::string t;
      bool s = false;
      if (type == "s") {
        char* end = nullptr;
        const long k = strtol(v.c_str(), &end, 10);
        if (!v.empty() && end && *end == 0 && k >= 0 && (size_t)k < sst.size()) t = sst[(size_t)k];  // (a bad index: empty)
        s = true;
      } else if (type == "inlineStr") { t = is; s = true; }
      else if (type == "str") { t = v; s = true; }
      else if (type == "b") t = v.empty() ? "" : v == "1" || v == "true" ? "TRUE" : "FALSE";
      else if (type == "e") t = v;
      else if (type == "d") t = v.substr(0, v.find("T00:00:00")) ;  // (an ISO date, strict files)
      else if (!v.empty()) {
        char* end = nullptr;
        const double d = strtod(v.c_str(), &end);
        t = end && *end == 0 ? value_text(d, style >= 0 && (size_t)style < xfs.size() ? xfs[(size_t)style] : Fmt{}, d1904) : v;
      }
      t = rdoc::tidy(t);
      if (!t.empty() && col <= 16384 && row <= 1048576) {
        if (g.cells >= budget) { g.cut = true; return false; }
        Cell& c = g.rows[row][col];
        g.cells += c.text.empty();
        c = {t, s};
      }
    } else if (n == "mergeCell" && merges.size() < 10000) {
      const std::string r = attr(tag, "ref");
      const size_t p = r.find(':');
      std::array<int, 4> m{};
      if (p != std::string::npos && cell_ref(r.substr(0, p), m[0], m[1]) && cell_ref(r.substr(p + 1), m[2], m[3]) && m[2] >= m[0] && m[3] >= m[1])
        merges.push_back(m);
    }
    return true;
  });
  // merged ranges: the value in the first cell only (some writers copy it into the others); bounded work
  size_t visits = 0;
  for (const auto& m : merges)
    for (auto r = g.rows.lower_bound(m[1]); r != g.rows.end() && r->first <= m[3] && visits < 4000000; ++r, ++visits)
      for (auto c = r->second.lower_bound(m[0]); c != r->second.end() && c->first <= m[2];) {
        if (r->first == m[1] && c->first == m[0]) { ++c; continue; }
        c = r->second.erase(c);
        --g.cells;
      }
  for (auto r = g.rows.begin(); r != g.rows.end();) r = r->second.empty() ? g.rows.erase(r) : std::next(r);
}

// a grid's rows as paragraphs: a header row (text in 2+ and most used columns, under up to 3 title lines) is repeated into
// each row as "Name: value"; without one, the values in their columns' order, an empty column as an empty field
inline void grid_paras(const Grid& g, rdoc::Doc& d) {
  std::map<int, int> pos;  // (used column -> its index among the used ones)
  for (const auto& [r, cols] : g.rows)
    for (const auto& [c, cell] : cols) pos[c] = 0;
  int k = 0;
  for (auto& [c, i] : pos) i = k++;
  if (g.rows.empty()) {
    d.paras.push_back({"(empty)", 1, 0});
    return;
  }
  auto it = g.rows.begin();
  for (int t = 0; t < 3 && it != g.rows.end() && pos.size() >= 3 && it->second.size() == 1; ++t, ++it)
    d.paras.push_back({it->second.begin()->second.text, 1, 0});
  std::map<int, std::string> head;
  if (it != g.rows.end() && std::next(it) != g.rows.end()) {
    bool ok = true;
    for (const auto& [c, cell] : it->second) ok = ok && cell.str && cell.text.size() <= 80;
    if (ok && it->second.size() >= 2 && it->second.size() * 10 >= pos.size() * 6) {
      std::string line;
      for (const auto& [c, cell] : it->second) {
        head[c] = cell.text;
        line += (line.empty() ? "" : " | ") + cell.text;
      }
      d.paras.push_back({line, 1, 0});
      ++it;
    }
  }
  for (; it != g.rows.end(); ++it) {
    std::vector<std::string> fields;
    int prev = -1;
    for (const auto& [c, cell] : it->second) {
      if (!head.empty()) {
        auto h = head.find(c);
        fields.push_back((h == head.end() ? "" : h->second + ": ") + cell.text);
        continue;
      }
      const int at = pos[c];
      fields.resize(fields.size() + (size_t)std::min(at - prev - 1, 8));  // (empty columns between: at most 8 shown)
      fields.push_back(cell.text);
      prev = at;
    }
    std::string line;
    for (size_t f = 0; f < fields.size(); ++f) line += (f ? " | " : "") + fields[f];
    d.paras.push_back({line, 1, 0});
  }
}

constexpr size_t kMaxCells = 200000;

// ---------------------------------------------------------------- writing
// a cell's text as a number: v and its number format code ("": General); false: it stays text. group: the thousands
// separator when a text cannot tell ('.': Indonesian "1.500" = 1500, ',': US "1,500" = 1500); "1.234,5", "1,234.5",
// "1.234.567", "2,5" and "3.14" tell by themselves
inline bool number(const std::string& in, double& v, std::string& fmt, char group = '.') {
  std::string t;
  for (size_t i = 0; i < in.size(); ++i) {  // (no-break and narrow no-break spaces as spaces)
    if (in.compare(i, 2, "\xC2\xA0") == 0) { t += ' '; i += 1; }
    else if (in.compare(i, 3, "\xE2\x80\xAF") == 0) { t += ' '; i += 2; }
    else t += in[i];
  }
  t = doc::trim(t);
  bool neg = false, plus = false;
  std::string cur;
  auto sign = [&] {
    if (!t.empty() && (t[0] == '-' || t[0] == '+')) { neg = t[0] == '-'; plus = t[0] == '+'; t = doc::trim(t.substr(1)); }
    else if (t.compare(0, 3, "\xE2\x88\x92") == 0) { neg = true; t = doc::trim(t.substr(3)); }  // (U+2212 minus)
  };
  auto currency = [&] {
    if (!cur.empty()) return;
    const std::string l = doc::lower(t.substr(0, 3));
    if (l == "idr") { cur = "IDR"; t = t.substr(3); }
    else if (l.rfind("rp", 0) == 0) { cur = "Rp"; t = t.substr(2); if (!t.empty() && t[0] == '.') t = t.substr(1); }
    else if (l.rfind("$", 0) == 0) { cur = "$"; t = t.substr(1); }
    t = doc::trim(t);
  };
  sign();
  currency();
  if (!neg) sign();
  bool pct = false;
  if (!t.empty() && t.back() == '%') { pct = true; t.pop_back(); t = doc::trim(t); }
  if ((pct && !cur.empty()) || t.empty() || !isdigit((unsigned char)t[0]) || !isdigit((unsigned char)t.back())) return false;
  size_t dots = 0, commas = 0;
  for (char c : t) {
    if (c == '.') ++dots;
    else if (c == ',') ++commas;
    else if (!isdigit((unsigned char)c)) return false;
  }
  char dec = 0, grp = 0;
  if (dots && commas) {
    dec = t.rfind('.') > t.rfind(',') ? '.' : ',';
    grp = dec == '.' ? ',' : '.';
    if ((dec == '.' ? dots : commas) > 1) return false;
  } else if (dots + commas > 1) grp = dots ? '.' : ',';
  else if (dots + commas == 1) {
    const char s = dots ? '.' : ',';
    const size_t p = t.find(s);
    const bool three = t.size() - p - 1 == 3 && p <= 3 && t[0] != '0';
    if (three && (s == group || !cur.empty())) grp = s;  // (with a currency always thousands: no cents in Rp)
    else dec = s;
  }
  std::string ip = dec ? t.substr(0, t.find(dec)) : t, fp = dec ? t.substr(t.find(dec) + 1) : "";
  if (grp) {  // (groups: 1-3 digits, then 3 each)
    std::string digits;
    for (size_t a = 0, first = 1;; first = 0) {
      const size_t e = ip.find(grp, a);
      const std::string part = ip.substr(a, e == std::string::npos ? std::string::npos : e - a);
      if (part.empty() || (first ? part.size() > 3 : part.size() != 3)) return false;
      digits += part;
      if (e == std::string::npos) break;
      a = e + 1;
    }
    ip = digits;
  }
  if ((ip.size() > 1 && ip[0] == '0' && !grp) || ip.size() + fp.size() > 15) return false;  // (an ID, a phone number)
  // (+6281234567890, or 12+ plain digits: a phone or account number, which Excel would show as 6.28E+12)
  if ((plus && !grp && fp.empty() && ip.size() >= 8) || (!grp && !dec && cur.empty() && !pct && ip.size() >= 12)) return false;
  v = strtod((ip + (fp.empty() ? "" : "." + fp)).c_str(), nullptr) * (neg ? -1 : 1);
  const std::string decimals = fp.empty() ? "" : "." + std::string(std::min<size_t>(fp.size(), 10), '0');
  if (pct) {
    v /= 100;
    fmt = "0" + decimals + "%";
  } else if (grp || !cur.empty()) {
    fmt = "#,##0" + decimals;
    if (!cur.empty()) fmt = "\"" + cur + (cur == "$" ? "" : " ") + "\"" + fmt;
  } else fmt.clear();
  return true;
}
// the thousands separator a number's text shows: '.' (Indonesian: "1.234,5", "1.234.567", "2,5"), ',' (US:
// "1,234.5", "1,234,567", "3.14"), 0 when it cannot tell ("1.500", "12")
inline char group_of(const std::string& t) {
  size_t dots = 0, commas = 0;
  for (char c : t) dots += c == '.', commas += c == ',';
  if (dots && commas) return t.rfind('.') > t.rfind(',') ? ',' : '.';
  if (dots > 1 || commas > 1) return dots ? '.' : ',';
  if (dots + commas == 1) {
    const size_t p = t.find(dots ? '.' : ',');
    size_t n = 0;
    while (p + 1 + n < t.size() && isdigit((unsigned char)t[p + 1 + n])) ++n;
    if (n != 3) return dots ? ',' : '.';  // (a decimal point: US; a decimal comma: Indonesian)
  }
  return 0;
}

// "2026-10-27", "27/10/2026", "27-10-2026" (an optional " hh:mm[:ss]", "T" after an ISO date) as an Excel serial date
inline bool date(const std::string& in, double& v, std::string& fmt, bool month_first = false) {
  const std::string t = doc::trim(in);
  size_t p = 0;
  auto num = [&](int& out, size_t lo, size_t hi) {
    const size_t s = p;
    out = 0;
    while (p < t.size() && isdigit((unsigned char)t[p]) && p - s < hi) out = out * 10 + (t[p++] - '0');
    return p - s >= lo;
  };
  auto lit = [&](char c) { return p < t.size() && t[p] == c ? (++p, true) : false; };
  int y = 0, m = 0, d = 0, hh = 0, mm = 0, ss = -1;
  const bool iso = t.size() >= 8 && t[4] == '-';
  if (iso) {
    if (!num(y, 4, 4) || !lit('-') || !num(m, 1, 2) || !lit('-') || !num(d, 1, 2)) return false;
  } else {
    if (!num(d, 1, 2) || p >= t.size() || (t[p] != '/' && t[p] != '-')) return false;
    const char sep = t[p++];
    if (!num(m, 1, 2) || !lit(sep) || !num(y, 4, 4)) return false;
    if (month_first && sep == '/') std::swap(d, m);  // (a US table: 03/04/2026 is March 4)
  }
  bool time = false;
  if (p < t.size()) {
    if (!lit(' ') && !(iso && lit('T'))) return false;
    while (lit(' ')) {}
    if (!num(hh, 1, 2) || !lit(':') || !num(mm, 2, 2)) return false;
    if (lit(':') && !num(ss, 2, 2)) return false;
    if (p != t.size() || hh > 23 || mm > 59 || ss > 59) return false;
    time = true;
  }
  static const int mdays[12] = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  const bool leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
  if (y < 1900 || y > 9999 || m < 1 || m > 12 || d < 1 || d > mdays[m - 1] || (m == 2 && d == 29 && !leap)) return false;
  const long long serial = days_from_civil(y, m, d) + 25569;
  if (serial < 61) return false;  // (before 1900-03-01: Excel's calendar is off by a day there)
  v = (double)serial + (hh * 3600 + mm * 60 + std::max(ss, 0)) / 86400.0;
  fmt = std::string(iso ? "yyyy-mm-dd" : "dd/mm/yyyy") + (time ? ss >= 0 ? " hh:mm:ss" : " hh:mm" : "");
  return true;
}

// text cut to n UTF-16 units (a sheet name's limit is 31), never inside a character
inline std::string cut_units(const std::string& s, size_t n) {
  size_t units = 0, i = 0;
  while (i < s.size()) {
    const unsigned char c = (unsigned char)s[i];
    const size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4, u = len == 4 ? 2 : 1;
    if (units + u > n) break;
    units += u;
    i += len;
  }
  return s.substr(0, std::min(i, s.size()));
}
// a valid sheet name not taken yet (case-insensitive): no [ ] : * ? / \, no ' at either end, 31 units at most
inline std::string sheet_name(const std::string& in, std::vector<std::string>& taken, const std::string& fallback) {
  auto clean = [](std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\'')) s.pop_back();
    while (!s.empty() && (s[0] == ' ' || s[0] == '\'')) s.erase(0, 1);
    return s;
  };
  std::string s;
  for (char c : in)
    if ((unsigned char)c >= 0x20 && !strchr("[]:*?/\\", c)) s += c;
  s = clean(cut_units(clean(rdoc::tidy(s)), 31));
  if (s.empty()) s = fallback;
  if (doc::lower(s) == "history") s += " (1)";  // (Excel's own name)
  const std::string base = s;
  for (int k = 2; std::find(taken.begin(), taken.end(), doc::lower(s)) != taken.end(); ++k) {
    const std::string suf = " (" + std::to_string(k) + ")";
    s = clean(cut_units(base, 31 - suf.size())) + suf;
  }
  taken.push_back(doc::lower(s));
  return s;
}

inline std::string plain(const std::vector<doc::Run>& runs) {
  std::string o;
  for (const doc::Run& r : runs) o += r.br ? " " : r.text;
  return rdoc::tidy(o);
}
inline size_t chars(const std::string& s) {
  size_t n = 0;
  for (unsigned char c : s) n += (c & 0xC0) != 0x80;
  return n;
}
inline std::string col_name(int c) {  // (0-based -> "A", "AB")
  std::string s;
  for (++c; c > 0; c = (c - 1) / 26) s.insert(s.begin(), (char)('A' + (c - 1) % 26));
  return s;
}

struct WCell {
  std::string text;  // (the source text: a string's value, a number's width)
  double v = 0;
  bool num = false, bold = false;
  std::string fmt;
};
struct WSheet {
  std::string name;
  std::vector<std::vector<WCell>> rows;
  std::vector<int> row_style;  // (Notes: 0 text, 1 a heading, 2 the title)
  bool table = true;
};

// the workbook: Markdown tables as sheets, the text around them on a first "Notes" sheet. numbers: "id" (1.234,5),
// "us" (1,234.5) or "auto": each table by the numbers in it that show their style ("1.234,5", "2,5", "3.14" ...),
// the whole document's when the table's tie or show none, Indonesian when nothing shows
inline std::string write(const std::string& md, const doc::Options& o, const std::string& numbers = "auto") {
  const std::vector<doc::Block> blocks = doc::parse(md);
  bool tables = false, text = false;
  auto votes = [](const doc::Block& b, int& id, int& us) {
    for (size_t r = 1; r < b.rows.size(); ++r)
      for (const auto& runs : b.rows[r]) {
        const std::string t = plain(runs);
        double x;
        std::string f;
        if (t.empty() || date(t, x, f) || !number(t, x, f)) continue;
        const char g = group_of(t);
        id += g == '.';
        us += g == ',';
      }
  };
  int doc_id = 0, doc_us = 0;
  for (const doc::Block& b : blocks) {
    tables = tables || b.kind == doc::Block::Table;
    text = text || b.kind == doc::Block::Para || b.kind == doc::Block::Bullet || b.kind == doc::Block::Number || b.kind == doc::Block::Code;
    if (b.kind == doc::Block::Table) votes(b, doc_id, doc_us);
  }
  const std::string nf = doc::lower(numbers);
  const char forced = nf == "id" ? '.' : nf == "us" || nf == "en" ? ',' : 0, doc_group = doc_us > doc_id ? ',' : '.';
  const bool has_notes = text || !tables;
  std::vector<std::string> taken;
  std::vector<WSheet> sheets;
  if (has_notes) {
    sheets.push_back({sheet_name("Notes", taken, "Notes"), {}, {}, false});
    auto add = [&](const std::string& s, int style) {
      if (!s.empty()) { sheets[0].rows.push_back({WCell{s}}); sheets[0].row_style.push_back(style); }
    };
    add(rdoc::tidy(o.title), 2);
    add(rdoc::tidy(o.subtitle), 1);
    add(rdoc::tidy(o.author + (o.author.empty() || o.date.empty() ? "" : ", ") + o.date), 0);
  }
  std::string heading;
  for (const doc::Block& b : blocks) {
    auto note = [&](const std::string& s, int style) {
      if (has_notes) { sheets[0].rows.push_back({WCell{s}}); sheets[0].row_style.push_back(style); }
    };
    if (b.kind == doc::Block::Heading) {
      heading = plain(b.runs);
      note(heading, 1);
    } else if (b.kind == doc::Block::Table) {
      WSheet s;
      s.name = sheet_name(heading, taken, "Sheet" + std::to_string(sheets.size() + !has_notes));
      heading.clear();
      char group = forced;
      if (!group) {
        int id = 0, us = 0;
        votes(b, id, us);
        group = us > id ? ',' : id > us ? '.' : doc_group;
      }
      for (size_t r = 0; r < b.rows.size(); ++r) {
        std::vector<WCell> row;
        for (const auto& runs : b.rows[r]) {
          WCell c;
          c.text = plain(runs);
          c.bold = r == 0 || (!runs.empty() && std::all_of(runs.begin(), runs.end(), [](const doc::Run& x) { return x.st.b || doc::trim(x.text).empty(); }));
          if (r > 0 && !c.text.empty()) c.num = date(c.text, c.v, c.fmt, group == ',') || number(c.text, c.v, c.fmt, group);
          row.push_back(c);
        }
        s.rows.push_back(row);
      }
      note("table: sheet '" + s.name + "'", 0);
      sheets.push_back(std::move(s));
    } else if (b.kind == doc::Block::Para) note(plain(b.runs), 0);
    else if (b.kind == doc::Block::Bullet || b.kind == doc::Block::Number)
      note(std::string((size_t)b.level * 4, ' ') + (b.kind == doc::Block::Bullet ? "\xE2\x80\xA2 " : std::to_string(b.num) + ". ") + plain(b.runs), 0);
    else if (b.kind == doc::Block::Code)
      for (const std::string& l : b.lines) note(l, 0);
  }

  // styles: fonts 0 normal, 1 bold, 2 bold and larger; fills 0 none, 1 gray125 (both required), 2 the header's;
  // borders 0 none, 1 the header's bottom line
  struct Xf { int fmt, font, fill, border; bool wrap; };
  std::vector<Xf> xfs = {{0, 0, 0, 0, false}};
  std::vector<std::pair<int, std::string>> numfmts;
  auto xf = [&](const std::string& code, int font, int fill, int border, bool wrap) {
    static const std::pair<const char*, int> builtin[] = {{"0", 1}, {"0.00", 2}, {"#,##0", 3}, {"#,##0.00", 4}, {"0%", 9}, {"0.00%", 10}};
    int id = code.empty() ? 0 : -1;
    for (const auto& [c, n] : builtin)
      if (code == c) id = n;
    for (const auto& [n, c] : numfmts)
      if (id < 0 && c == code) id = n;
    if (id < 0) { id = 164 + (int)numfmts.size(); numfmts.push_back({id, code}); }
    for (size_t k = 0; k < xfs.size(); ++k)
      if (xfs[k].fmt == id && xfs[k].font == font && xfs[k].fill == fill && xfs[k].border == border && xfs[k].wrap == wrap) return (int)k;
    xfs.push_back({id, font, fill, border, wrap});
    return (int)xfs.size() - 1;
  };
  const std::string X = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n";
  const std::string R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
  const std::string NS = "xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"" + R + "\"";
  std::vector<std::string> sst;
  std::map<std::string, int> sst_at;
  size_t sst_count = 0;
  auto numv = [](double v) {
    char b[40];
    snprintf(b, sizeof b, "%.15g", v);
    if (strtod(b, nullptr) != v) snprintf(b, sizeof b, "%.17g", v);
    return std::string(b);
  };
  auto quote = [](const std::string& name) {  // ('Name' in a formula, ' doubled)
    std::string q = "'";
    for (char c : name) q += c == '\'' ? std::string("''") : std::string(1, c);
    return q + "'";
  };
  size_t active = has_notes && sheets.size() > 1 ? 1 : 0;
  std::vector<std::pair<std::string, std::string>> parts;
  std::string defined;
  for (size_t si = 0; si < sheets.size(); ++si) {
    const WSheet& s = sheets[si];
    size_t ncols = 0;
    for (const auto& r : s.rows) ncols = std::max(ncols, r.size());
    std::vector<double> width(ncols, 8);
    std::string data;
    for (size_t r = 0; r < s.rows.size(); ++r) {
      data += "<row r=\"" + std::to_string(r + 1) + "\">";
      for (size_t c = 0; c < s.rows[r].size(); ++c) {
        const WCell& cell = s.rows[r][c];
        if (cell.text.empty()) continue;
        const size_t len = chars(cell.text);
        int style = 0;
        if (!s.table) {
          const int rs = s.row_style[r];
          style = xf("", rs == 2 ? 2 : rs == 1 ? 1 : 0, 0, 0, true);
        } else if (r == 0) style = xf("", 1, 2, 1, false);
        else style = xf(cell.num ? cell.fmt : "", cell.bold ? 1 : 0, 0, 0, !cell.num && len > 60);
        width[c] = std::max(width[c], std::min(60.0, len * 1.1 + 2 + (s.table && r == 0 ? 3 : 0)));
        data += "<c r=\"" + col_name((int)c) + std::to_string(r + 1) + "\"" + (style ? " s=\"" + std::to_string(style) + "\"" : "");
        if (cell.num) data += "><v>" + numv(cell.v) + "</v></c>";
        else {
          auto it = sst_at.find(cell.text);
          if (it == sst_at.end()) {
            it = sst_at.emplace(cell.text, (int)sst.size()).first;
            sst.push_back(cell.text);
          }
          ++sst_count;
          data += " t=\"s\"><v>" + std::to_string(it->second) + "</v></c>";
        }
      }
      data += "</row>";
    }
    if (!s.table && ncols) width[0] = 100;
    const std::string last = ncols && !s.rows.empty() ? col_name((int)ncols - 1) + std::to_string(s.rows.size()) : "A1";
    const std::string ref = last == "A1" ? "A1" : "A1:" + last;
    std::string w = X + "<worksheet " + NS + "><dimension ref=\"" + ref + "\"/><sheetViews><sheetView" + (si == active ? " tabSelected=\"1\"" : "") +
                    " workbookViewId=\"0\">";
    if (s.table && !s.rows.empty())
      w += "<pane ySplit=\"1\" topLeftCell=\"A2\" activePane=\"bottomLeft\" state=\"frozen\"/><selection pane=\"bottomLeft\" activeCell=\"A2\" sqref=\"A2\"/>";
    w += "</sheetView></sheetViews><sheetFormatPr defaultRowHeight=\"15\"/>";
    if (ncols) {
      w += "<cols>";
      for (size_t c = 0; c < ncols; ++c) {
        char b[24];
        snprintf(b, sizeof b, "%.1f", width[c]);
        w += "<col min=\"" + std::to_string(c + 1) + "\" max=\"" + std::to_string(c + 1) + "\" width=\"" + b + "\" customWidth=\"1\"/>";
      }
      w += "</cols>";
    }
    w += data.empty() ? "<sheetData/>" : "<sheetData>" + data + "</sheetData>";
    if (s.table && ncols && !s.rows.empty()) {
      w += "<autoFilter ref=\"" + ref + "\"/>";
      std::string abs = "$A$1";
      if (ref != "A1") abs += ":$" + col_name((int)ncols - 1) + "$" + std::to_string(s.rows.size());
      defined += "<definedName name=\"_xlnm._FilterDatabase\" localSheetId=\"" + std::to_string(si) + "\" hidden=\"1\">" +
                 doc::xesc(quote(s.name) + "!" + abs) + "</definedName>";
    }
    w += "<pageMargins left=\"0.7\" right=\"0.7\" top=\"0.75\" bottom=\"0.75\" header=\"0.3\" footer=\"0.3\"/></worksheet>";
    parts.push_back({"xl/worksheets/sheet" + std::to_string(si + 1) + ".xml", w});
  }

  std::string types = X + "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
                          "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
                          "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
                          "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>";
  std::string sheets_xml, rels = X + "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">";
  for (size_t si = 0; si < sheets.size(); ++si) {
    const std::string n = std::to_string(si + 1);
    types += "<Override PartName=\"/xl/worksheets/sheet" + n + ".xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>";
    sheets_xml += "<sheet name=\"" + doc::xesc(sheets[si].name) + "\" sheetId=\"" + n + "\" r:id=\"rId" + n + "\"/>";
    rels += "<Relationship Id=\"rId" + n + "\" Type=\"" + R + "/worksheet\" Target=\"worksheets/sheet" + n + ".xml\"/>";
  }
  const std::string ns = std::to_string(sheets.size() + 1), nt = std::to_string(sheets.size() + 2);
  rels += "<Relationship Id=\"rId" + ns + "\" Type=\"" + R + "/styles\" Target=\"styles.xml\"/>";
  if (!sst.empty()) rels += "<Relationship Id=\"rId" + nt + "\" Type=\"" + R + "/sharedStrings\" Target=\"sharedStrings.xml\"/>";
  rels += "</Relationships>";
  types += "<Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>";
  if (!sst.empty()) types += "<Override PartName=\"/xl/sharedStrings.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml\"/>";
  types += "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
           "<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/></Types>";

  // styles.xml in the schema's order: numFmts, fonts, fills, borders, cellStyleXfs, cellXfs, cellStyles
  char fsz[16];
  snprintf(fsz, sizeof fsz, "%g", std::clamp(o.font_size, 6.0, 36.0));
  const std::string font_name = doc::xesc(o.font.empty() ? std::string("Calibri") : o.font);
  std::string accent = doc::hex_color(o.accent_color);
  if (accent.empty()) accent = "DCE6F1";
  std::string st = X + "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">";
  if (!numfmts.empty()) {
    st += "<numFmts count=\"" + std::to_string(numfmts.size()) + "\">";
    for (const auto& [n, c] : numfmts) st += "<numFmt numFmtId=\"" + std::to_string(n) + "\" formatCode=\"" + doc::xesc(c) + "\"/>";
    st += "</numFmts>";
  }
  char big[16];
  snprintf(big, sizeof big, "%g", std::clamp(o.font_size, 6.0, 36.0) + 3);
  st += "<fonts count=\"3\"><font><sz val=\"" + std::string(fsz) + "\"/><name val=\"" + font_name + "\"/><family val=\"2\"/></font>"
        "<font><b/><sz val=\"" + std::string(fsz) + "\"/><name val=\"" + font_name + "\"/><family val=\"2\"/></font>"
        "<font><b/><sz val=\"" + std::string(big) + "\"/><name val=\"" + font_name + "\"/><family val=\"2\"/></font></fonts>"
        "<fills count=\"3\"><fill><patternFill patternType=\"none\"/></fill><fill><patternFill patternType=\"gray125\"/></fill>"
        "<fill><patternFill patternType=\"solid\"><fgColor rgb=\"FF" + accent + "\"/><bgColor indexed=\"64\"/></patternFill></fill></fills>"
        "<borders count=\"2\"><border><left/><right/><top/><bottom/><diagonal/></border>"
        "<border><left/><right/><top/><bottom style=\"thin\"><color rgb=\"FF808080\"/></bottom><diagonal/></border></borders>"
        "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
        "<cellXfs count=\"" + std::to_string(xfs.size()) + "\">";
  for (const Xf& x : xfs) {
    st += "<xf numFmtId=\"" + std::to_string(x.fmt) + "\" fontId=\"" + std::to_string(x.font) + "\" fillId=\"" + std::to_string(x.fill) +
          "\" borderId=\"" + std::to_string(x.border) + "\" xfId=\"0\"" + (x.fmt ? " applyNumberFormat=\"1\"" : "") +
          (x.font ? " applyFont=\"1\"" : "") + (x.fill ? " applyFill=\"1\"" : "") + (x.border ? " applyBorder=\"1\"" : "");
    st += x.wrap ? " applyAlignment=\"1\"><alignment vertical=\"top\" wrapText=\"1\"/></xf>" : "/>";
  }
  st += "</cellXfs><cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles></styleSheet>";

  std::string ss = X + "<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" count=\"" + std::to_string(sst_count) +
                   "\" uniqueCount=\"" + std::to_string(sst.size()) + "\">";
  for (const std::string& s : sst) {
    const bool keep = s.front() == ' ' || s.back() == ' ' || s.find("  ") != std::string::npos;
    ss += std::string("<si><t") + (keep ? " xml:space=\"preserve\"" : "") + ">" + doc::xesc(s) + "</t></si>";
  }
  ss += "</sst>";

  std::vector<std::pair<std::string, std::string>> out;
  out.push_back({"[Content_Types].xml", types});
  out.push_back({"_rels/.rels", X + "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                                    "<Relationship Id=\"rId1\" Type=\"" + R + "/officeDocument\" Target=\"xl/workbook.xml\"/>"
                                    "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>"
                                    "<Relationship Id=\"rId3\" Type=\"" + R + "/extended-properties\" Target=\"docProps/app.xml\"/></Relationships>"});
  out.push_back({"docProps/core.xml",
                 X + "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
                     "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:dcterms=\"http://purl.org/dc/terms/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
                     "<dc:title>" + doc::xesc(o.title) + "</dc:title><dc:creator>" + doc::xesc(o.author.empty() ? std::string("arcinferzero") : o.author) +
                     "</dc:creator></cp:coreProperties>"});
  out.push_back({"docProps/app.xml", X + "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\">"
                                         "<Application>arcinferzero</Application></Properties>"});
  // workbook.xml in the schema's order: workbookPr, bookViews, sheets, definedNames
  out.push_back({"xl/workbook.xml", X + "<workbook " + NS + "><workbookPr/><bookViews><workbookView activeTab=\"" + std::to_string(active) +
                                        "\"/></bookViews><sheets>" + sheets_xml + "</sheets>" +
                                        (defined.empty() ? "" : "<definedNames>" + defined + "</definedNames>") + "</workbook>"});
  out.push_back({"xl/_rels/workbook.xml.rels", rels});
  out.push_back({"xl/styles.xml", st});
  if (!sst.empty()) out.push_back({"xl/sharedStrings.xml", ss});
  for (auto& p : parts) out.push_back(std::move(p));
  return doc::zip(out);
}

}  // namespace xlsx

namespace rdoc {

inline Doc read_xlsx(const std::string& zip) {
  Doc d;
  if (zip.compare(0, 4, "\xd0\xcf\x11\xe0") == 0) {
    d.note = "an Excel 97-2003 .xls (binary): save it as XLSX to read it";
    return d;
  }
  std::string wb, rels, sx, stx;
  if (!infl::zip_entry(zip, "xl/workbook.xml", wb, 16u << 20)) {
    d.note = "not a readable XLSX (no xl/workbook.xml)";
    return d;
  }
  infl::zip_entry(zip, "xl/_rels/workbook.xml.rels", rels, 4u << 20);
  std::map<std::string, std::string> target;  // (relationship id -> part)
  xlsx::walk(rels, [&](const std::string& n, const std::string& tag, size_t, size_t) {
    if (n == "Relationship") {
      std::string t = xlsx::attr(tag, "Target");
      t = !t.empty() && t[0] == '/' ? t.substr(1) : "xl/" + t;
      target[xlsx::attr(tag, "Id")] = t;
    }
    return target.size() < 100000;
  });
  struct Sheet { std::string name, part; bool hidden; };
  std::vector<Sheet> sheets;
  bool d1904 = false;
  xlsx::walk(wb, [&](const std::string& n, const std::string& tag, size_t, size_t) {
    if (n == "workbookPr") d1904 = xlsx::attr(tag, "date1904") == "1" || xlsx::attr(tag, "date1904") == "true";
    else if (n == "sheet") {
      std::string id = xlsx::attr(tag, "r:id");
      for (size_t p = tag.find(":id="); id.empty() && p != std::string::npos; p = std::string::npos) {  // (another prefix)
        const size_t s = tag.rfind(' ', p);
        if (s != std::string::npos) id = xlsx::attr(tag, tag.substr(s + 1, p + 3 - s - 1).c_str());
      }
      auto t = target.find(id);
      sheets.push_back({xlsx::attr(tag, "name"), t != target.end() ? t->second : "xl/worksheets/sheet" + std::to_string(sheets.size() + 1) + ".xml",
                        !xlsx::attr(tag, "state").empty() && xlsx::attr(tag, "state") != "visible"});
    }
    return sheets.size() < 1000;
  });
  std::vector<std::string> sst;
  if (infl::zip_entry(zip, "xl/sharedStrings.xml", sx, 64u << 20) || !sx.empty()) sst = xlsx::shared_strings(sx);
  sx.clear();
  std::vector<xlsx::Fmt> xfs;
  if (infl::zip_entry(zip, "xl/styles.xml", stx, 16u << 20)) {
    std::map<int, std::string> codes;
    bool in_xfs = false;
    xlsx::walk(stx, [&](const std::string& n, const std::string& tag, size_t, size_t) {
      if (n == "numFmt") codes[atoi(xlsx::attr(tag, "numFmtId").c_str())] = xlsx::attr(tag, "formatCode");
      else if (n == "cellXfs") in_xfs = !xlsx::self_closing(tag);
      else if (n == "/cellXfs") in_xfs = false;
      else if (n == "xf" && in_xfs) {
        const int id = atoi(xlsx::attr(tag, "numFmtId").c_str());
        auto c = codes.find(id);
        xfs.push_back(xlsx::fmt_of(id, c == codes.end() ? "" : c->second));
      }
      return xfs.size() < 100000;
    });
  }
  size_t budget = xlsx::kMaxCells, inflated = 0;
  std::vector<std::string> cut, unread;
  for (const Sheet& s : sheets) {
    d.paras.push_back({"Sheet: " + rdoc::tidy(s.name) + (s.hidden ? " (hidden)" : ""), 1, 1});
    if (!budget || inflated > (512u << 20)) {  // (the cells, and all sheet parts together: a bounded read)
      unread.push_back(s.name);
      d.paras.push_back({budget ? "(not read: the workbook is too large to read whole)" : "(not read: the cell limit was reached)", 1, 0});
      continue;
    }
    std::string x;
    const bool whole = infl::zip_entry(zip, s.part, x, 64u << 20);  // (a part cut by the limit: what came before it)
    inflated += x.size();
    xlsx::Grid g;
    g.name = s.name;
    xlsx::read_sheet(x, sst, xfs, d1904, g, budget);
    budget -= g.cells;
    if (g.cut || (!whole && !x.empty())) cut.push_back(s.name);
    xlsx::grid_paras(g, d);
  }
  if (sheets.empty()) d.note = "the workbook has no sheets";
  else if (!cut.empty() || !unread.empty())
    d.note = "large workbook: only the first " + std::to_string(xlsx::kMaxCells) + " cells were read (cut in sheet '" +
             (cut.empty() ? unread[0] : cut[0]) + "'" + (unread.empty() ? "" : ", " + std::to_string(unread.size()) + " sheet(s) after it not read") + ")";
  return d;
}

// CSV / TSV: fields in quotes may hold the delimiter, "" and line breaks; delim 0: , ; or tab, whichever the first
// line has most (outside quotes)
inline Doc read_csv(const std::string& data, char delim) {
  Doc d;
  std::string t = data;
  if (t.size() >= 3 && (unsigned char)t[0] == 0xEF && (unsigned char)t[1] == 0xBB && (unsigned char)t[2] == 0xBF) t.erase(0, 3);
  if (!valid_utf8(t)) {  // (an ANSI file: Windows-1252)
    std::string u;
    for (unsigned char c : t) put_utf8(u, cp1252(c));
    t.swap(u);
  }
  if (!delim) {
    size_t n[3] = {};
    bool q = false;
    for (size_t i = 0; i < t.size() && (q || t[i] != '\n'); ++i) {
      if (t[i] == '"') q = !q;
      else if (!q) n[0] += t[i] == ',', n[1] += t[i] == ';', n[2] += t[i] == '\t';
    }
    delim = n[1] > n[0] && n[1] >= n[2] ? ';' : n[2] > n[0] ? '\t' : ',';
  }
  xlsx::Grid g;
  int row = 1, col = 1;
  std::string f;
  bool q = false;
  auto put = [&] {
    const std::string v = tidy(f);
    if (!v.empty() && !g.cut) {
      if (g.cells >= xlsx::kMaxCells) g.cut = true;
      else {
        double x;
        std::string fmt;
        g.rows[row][col] = {v, !(xlsx::date(v, x, fmt) || xlsx::number(v, x, fmt))};
        ++g.cells;
      }
    }
    f.clear();
    ++col;
  };
  for (size_t i = 0; i < t.size() && !g.cut; ++i) {
    const char c = t[i];
    if (q) {
      if (c == '"' && i + 1 < t.size() && t[i + 1] == '"') { f += '"'; ++i; }
      else if (c == '"') q = false;
      else f += c;
    } else if (c == '"' && tidy(f).empty()) { q = true; f.clear(); }
    else if (c == delim) put();
    else if (c == '\n' || c == '\r') {
      if (c == '\r' && i + 1 < t.size() && t[i + 1] == '\n') ++i;
      put();
      ++row;
      col = 1;
    } else f += c;
  }
  put();
  if (g.rows.empty()) {
    d.note = "the file is empty";
    return d;
  }
  xlsx::grid_paras(g, d);
  if (g.cut) d.note = "large file: only the first " + std::to_string(xlsx::kMaxCells) + " cells were read";
  return d;
}

}  // namespace rdoc
