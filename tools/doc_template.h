// Word templates: a .docx the user keeps (a company's minutes form, with its letterhead, logo, fonts and signature
// block) filled with data. Only the text of its table cells changes: the first table's "Label | : | value" rows by
// their label, every other table's rows rebuilt from its first data row (one row per data row given); a table given
// no data stays as it is (the signature block). Everything else in the file (header, footer, images, styles, the
// paragraphs between the tables) is copied unchanged. Used by POST /v1/tools/minutes_docx (server_file_tools.h).
#pragma once
#include "doc_writer.h"  // doc::zip, doc::xesc
#include "inflate.h"
#include "../src/json.h"
#include <string>
#include <vector>

namespace doct {

// the names in a zip's central directory, in order
inline std::vector<std::string> zip_names(const std::string& zip) {
  std::vector<std::string> names;
  const unsigned char* z = (const unsigned char*)zip.data();
  auto u16 = [&](size_t o) { return o + 2 <= zip.size() ? (unsigned)(z[o] | z[o + 1] << 8) : 0u; };
  auto u32 = [&](size_t o) { return o + 4 <= zip.size() ? (uint32_t)(z[o] | z[o + 1] << 8 | z[o + 2] << 16 | (uint32_t)z[o + 3] << 24) : 0u; };
  if (zip.size() < 22) return names;
  size_t e = zip.size() - 22;
  const size_t stop = zip.size() > 65557 ? zip.size() - 65557 : 0;
  while (u32(e) != 0x06054b50) {
    if (e == stop) return names;
    --e;
  }
  size_t cd = u32(e + 16);
  for (unsigned k = 0, n = u16(e + 10); k < n && cd + 46 <= zip.size() && u32(cd) == 0x02014b50; ++k) {
    const unsigned nl = u16(cd + 28), xl = u16(cd + 30), cl = u16(cd + 32);
    if (cd + 46 + nl > zip.size()) break;
    names.push_back(zip.substr(cd + 46, nl));
    cd += 46 + nl + xl + cl;
  }
  return names;
}

struct Span { size_t a = 0, b = 0; };  // [a, b): an element with its tags
// the next <w:TAG ...>...</w:TAG> starting in [from, to) (nested ones of the same tag counted; <w:TAGPr> is another tag)
inline bool next_elem(const std::string& x, const std::string& tag, size_t from, size_t to, Span& s) {
  const std::string open = "<w:" + tag, close = "</w:" + tag + ">";
  auto is_open = [&](size_t p) {
    const char c = p + open.size() < x.size() ? x[p + open.size()] : 0;
    return c == '>' || c == ' ' || c == '/';
  };
  size_t p = x.find(open, from);
  while (p != std::string::npos && p < to && !is_open(p)) p = x.find(open, p + 1);
  if (p == std::string::npos || p >= to) return false;
  int depth = 0;
  for (size_t q = p; q < x.size();) {
    const size_t o = x.find(open, q), c = x.find(close, q);
    if (c == std::string::npos) return false;
    if (o != std::string::npos && o < c) {
      if (is_open(o)) ++depth;
      q = o + open.size();
    } else {
      if (--depth == 0) {
        s = {p, c + close.size()};
        return true;
      }
      q = c + close.size();
    }
  }
  return false;
}
// the direct children TAG of the element `in` (an element's own nested ones are skipped with it)
inline std::vector<Span> children(const std::string& x, Span in, const std::string& tag) {
  std::vector<Span> out;
  Span s;
  for (size_t from = in.a + 1; next_elem(x, tag, from, in.b, s); from = s.b) out.push_back(s);
  return out;
}
inline std::string unesc(std::string s) {
  const std::pair<const char*, const char*> ents[] = {{"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}, {"&amp;", "&"}};
  for (const auto& [e, c] : ents)
    for (size_t p = 0; (p = s.find(e, p)) != std::string::npos; p += strlen(c)) s.replace(p, strlen(e), c);
  return s;
}
// the text of an element: its <w:t> pieces, paragraphs joined by a line break
inline std::string text(const std::string& x, Span in) {
  std::string out;
  for (const Span& p : children(x, in, "p").empty() ? std::vector<Span>{in} : children(x, in, "p")) {
    if (!out.empty()) out += "\n";
    for (size_t q = p.a; q < p.b;) {
      const size_t t = x.find("<w:t", q);
      if (t == std::string::npos || t >= p.b) break;
      const char c = x[t + 4];
      if (c != '>' && c != ' ') { q = t + 4; continue; }
      const size_t gt = x.find('>', t), end = x.find("</w:t>", gt);
      if (gt == std::string::npos || end == std::string::npos || end > p.b) break;
      out += unesc(x.substr(gt + 1, end - gt - 1));
      q = end + 6;
    }
  }
  return out;
}
// every paragraph of a cell's text: nested paragraphs and tables of a cell are taken as text too
inline std::string cell_text(const std::string& x, Span tc) {
  std::string out;
  Span p;
  for (size_t from = tc.a + 1; next_elem(x, "p", from, tc.b, p); from = p.b) {
    if (!out.empty()) out += "\n";
    out += text(x, {p.a, p.b});
  }
  return out;
}
inline std::string trim(std::string s) {
  while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
  size_t i = 0;
  while (i < s.size() && isspace((unsigned char)s[i])) ++i;
  return s.substr(i);
}
inline std::string lower(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  return s;
}

// A cell with new text: its own properties kept, then one paragraph per line, each with the formatting of the cell's
// first paragraph and first run (a line "- x" or "• x" stays a line of its own; **bold** is not parsed: plain text)
inline std::string cell_with(const std::string& x, Span tc, const std::vector<std::string>& lines) {
  const std::string c = x.substr(tc.a, tc.b - tc.a);
  Span pr{0, 0}, p0{0, 0}, r0{0, 0};
  const bool has_pr = next_elem(c, "tcPr", 0, c.size(), pr);
  std::string ppr, rpr;
  if (next_elem(c, "p", 0, c.size(), p0)) {
    Span s;
    if (next_elem(c, "pPr", p0.a + 1, p0.b, s)) {
      ppr = c.substr(s.a, s.b - s.a);
      // (a run style inside the paragraph mark is kept; nothing else of pPr refers to the old text)
    }
    if (next_elem(c, "r", p0.a + 1, p0.b, r0) && next_elem(c, "rPr", r0.a + 1, r0.b, s)) rpr = c.substr(s.a, s.b - s.a);
  }
  const size_t gt = c.find('>');
  std::string out = c.substr(0, gt + 1);  // (<w:tc ...>)
  if (has_pr) out += c.substr(pr.a, pr.b - pr.a);
  for (const std::string& l : lines.empty() ? std::vector<std::string>{""} : lines) {
    out += "<w:p>" + ppr;
    if (!l.empty()) out += "<w:r>" + rpr + "<w:t xml:space=\"preserve\">" + doc::xesc(l) + "</w:t></w:r>";
    out += "</w:p>";
  }
  return out + "</w:tc>";
}
inline std::vector<std::string> lines_of(const Json& v) {
  std::vector<std::string> out;
  auto add = [&](const std::string& s) {
    size_t a = 0;
    for (size_t b; (b = s.find('\n', a)) != std::string::npos; a = b + 1) out.push_back(trim(s.substr(a, b - a)));
    out.push_back(trim(s.substr(a)));
  };
  if (v.is_array())
    for (const Json& e : v.items()) add(e.is_string() ? e.str() : e.dump());
  else if (v.is_string()) add(v.str());
  else if (!v.is_null()) add(v.dump());
  while (!out.empty() && out.back().empty()) out.pop_back();
  return out;
}

// a cell's text compared with a header's: lower case, spaces and the punctuation around it gone ("No." == "no",
// "Nomor" stays "nomor")
inline std::string norm_cell(const std::string& s) {
  std::string o;
  for (unsigned char c : s)
    if (isalnum(c) || c >= 0x80) o += (char)tolower(c);
  return o;
}
inline std::string lines_text(const Json& v) {
  std::string o;
  for (const std::string& l : lines_of(v)) o += l;
  return o;
}

// What a template asks for (the web UI's prompt to the model is made from it): its first table's labels ("fields"), and
// for every other table the paragraph just before it (its heading), its header row and its number of data rows.
inline Json structure(const std::string& docx, std::string& bad) {
  std::string x;
  if (!infl::zip_entry(docx, "word/document.xml", x, 64u << 20)) { bad = "not a readable .docx"; return Json(); }
  Span body;
  if (!next_elem(x, "body", 0, x.size(), body)) { bad = "no document body"; return Json(); }
  Json out = Json::object(), tables = Json::array();
  std::string last_para;
  size_t k = 0;
  for (size_t q = body.a + 1; q < body.b;) {
    Span p, t;
    const bool hp = next_elem(x, "p", q, body.b, p), ht = next_elem(x, "tbl", q, body.b, t);
    if (!hp && !ht) break;
    if (hp && (!ht || p.a < t.a)) {
      const std::string s = trim(text(x, p));
      if (!s.empty()) last_para = s;
      q = p.b;
      continue;
    }
    const std::vector<Span> rows = children(x, t, "tr");
    Json tj = Json::object();
    tj.set("heading", last_para);
    if (k == 0) {
      Json f = Json::array();
      for (const Span& r : rows) {
        const auto cells = children(x, r, "tc");
        if (!cells.empty()) f.push(trim(cell_text(x, cells[0])));
      }
      out.set("fields", f);
    } else {
      Json h = Json::array();
      if (!rows.empty())
        for (const Span& c : children(x, rows[0], "tc")) h.push(trim(cell_text(x, c)));
      tj.set("columns", h);
      tj.set("rows", (double)(rows.empty() ? 0 : rows.size() - 1));
      tables.push(tj);
    }
    last_para.clear();
    ++k;
    q = t.b;
  }
  out.set("tables", tables);
  return out;
}

// The template filled: data {"fields": {label: text or [lines]}, "tables": [table 2's rows, table 3's ...]}, each
// table's rows [[cell, cell, ...], ...] (a cell: text, lines joined by \n, or an array of lines); a table given null or
// left out stays as it is. "" with bad when the template cannot be read.
inline std::string fill(const std::string& docx, const Json& data, std::string& bad) {
  std::string x;
  if (!infl::zip_entry(docx, "word/document.xml", x, 64u << 20)) { bad = "not a readable .docx"; return ""; }
  Span body;
  if (!next_elem(x, "body", 0, x.size(), body)) { bad = "no document body"; return ""; }
  std::vector<Span> tbls;
  for (Span t; next_elem(x, "tbl", tbls.empty() ? body.a + 1 : tbls.back().b, body.b, t);) tbls.push_back(t);
  if (tbls.empty()) { bad = "the template has no table to fill"; return ""; }
  // (back to front: an edit moves only what comes after it)
  for (size_t k = tbls.size(); k-- > 0;) {
    const Span t = tbls[k];
    const std::vector<Span> rows = children(x, t, "tr");
    std::string nt;
    if (k == 0) {  // label rows: the last cell of a row whose first cell is a label given
      const Json& f = data["fields"];
      if (!f.is_object()) continue;
      nt = x.substr(t.a, (rows.empty() ? t.b : rows[0].a) - t.a);
      for (const Span& r : rows) {
        const auto cells = children(x, r, "tc");
        std::string row = x.substr(r.a, r.b - r.a);
        if (cells.size() >= 2) {
          const std::string label = lower(trim(cell_text(x, cells[0])));
          for (const auto& [key, val] : f.entries())
            if (lower(trim(key)) == label) {
              const Span last = cells.back();
              row = x.substr(r.a, last.a - r.a) + cell_with(x, last, lines_of(val)) + x.substr(last.b, r.b - last.b);
              break;
            }
        }
        nt += row;
      }
      nt += x.substr(rows.empty() ? t.b : rows.back().b, t.b - (rows.empty() ? t.b : rows.back().b));
    } else {
      const Json& rowsd = data["tables"][(int)(k - 1)];
      if (!rowsd.is_array() || rows.size() < 2) continue;
      const Span proto = rows[1];
      const auto pcells = children(x, proto, "tc");
      nt = x.substr(t.a, rows[1].a - t.a);  // (the table's properties and its header row)
      // (a data row that repeats the header ("No | Acara", "No. | Acara") is left out: models begin tables with it,
      //  and the form has the header already)
      std::vector<std::string> head;
      for (const Span& c : children(x, rows[0], "tc")) head.push_back(norm_cell(cell_text(x, c)));
      auto is_head = [&](const Json& rd) {
        if (!rd.is_array() || rd.items().empty()) return false;
        size_t same = 0, n = std::min(head.size(), rd.items().size());
        for (size_t c = 0; c < n; ++c) same += norm_cell(lines_text(rd[(int)c])) == head[c];
        const bool numbered = !head.empty() && (head[0] == "no" || head[0] == "nomor");
        const std::string first = norm_cell(lines_text(rd[0]));
        const bool first_head = first == "no" || first == "nomor" || first == head[0];  // (a row number is a digit)
        if (numbered && first_head) same += first != head[0];  // ("Nomor" over "No": the same header cell)
        return n > 0 && (same == n || (numbered && first_head && same * 2 >= n));
      };
      for (const Json& rd : rowsd.items()) {
        if (is_head(rd)) continue;
        std::string row = x.substr(proto.a, (pcells.empty() ? proto.b : pcells[0].a) - proto.a);
        for (size_t c = 0; c < pcells.size(); ++c) {
          row += cell_with(x, pcells[c], lines_of(rd[(int)c]));
          const size_t gap_end = c + 1 < pcells.size() ? pcells[c + 1].a : proto.b;
          row += x.substr(pcells[c].b, gap_end - pcells[c].b);
        }
        nt += row;
      }
      nt += x.substr(rows.back().b, t.b - rows.back().b);  // (</w:tbl>)
    }
    x = x.substr(0, t.a) + nt + x.substr(t.b);
  }
  // the zip again: every entry as it was, the document new
  std::vector<std::pair<std::string, std::string>> files;
  for (const std::string& n : zip_names(docx)) {
    if (n.empty() || n.back() == '/') continue;
    std::string d;
    if (n == "word/document.xml") d = x;
    else if (!infl::zip_entry(docx, n, d, 256u << 20)) { bad = "cannot read " + n + " of the template"; return ""; }
    files.push_back({n, std::move(d)});
  }
  return doc::zip(files);
}

}  // namespace doct
