// Documents the model writes (the web UI's create_file tool): Markdown in, a .docx or .pdf file out, written here
// with no library.
//
// The Markdown: # .. #### headings, paragraphs, **bold**, *italic*, `code`, ~~strike~~, ==highlight==, - / * / 1.
// lists (two spaces per level), ``` code blocks, | tables | (an alignment row :--- / :---: / ---: sets each column),
// --- rules; inline HTML: <b> <i> <u> <s> <mark> <br> and <span style="color: #c00000; background: #fff2cc;
// font-size: 14pt; font-weight: bold; font-style: italic; text-decoration: underline">; blocks between ::: center
// (or right, justify, left) and :::; \pagebreak (or \newpage, <!-- pagebreak -->) starts a new page.
// The document's options (Options): paper (A4, Letter, Legal) and orientation, margins, font (a name for DOCX; PDF
// uses Helvetica, or Times for serif names) and size, line spacing, heading / accent / text colors, header and footer
// text, page numbers, a title block (title, subtitle, author, date).
//
//   DOCX: a ZIP (stored, CRC-32) of WordprocessingML written in the schema's element order (Word repairs a file whose
//         elements are out of order), styles (Title, Subtitle, Heading1-4, Code, List Paragraph), numbering (bullets,
//         numbers restarting per list), tables with a grid, cell widths and a repeated header row, header and footer
//         parts with PAGE / NUMPAGES fields.
//   PDF:  PDF 1.4 with the standard fonts (Helvetica, Times and Courier families, WinAnsi: Latin-1 plus the usual
//         quotes, dashes, bullet, euro; other characters become '?'), lines laid out by the fonts' character widths
//         (left, center, right, justify), colors, highlight, underline / strike, tables with wrapped cells and a
//         repeated header row, header / footer text, page numbers.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace doc {

struct Style {
  bool b = false, i = false, u = false, s = false, code = false, mark = false;
  std::string color, bg;  // ("RRGGBB" or empty)
  double size = 0;        // (points; 0: the block's size)
};
struct Run {
  std::string text;  // (UTF-8)
  Style st;
  bool br = false;  // (a line break)
};
enum Align { Left = 0, Center = 1, Right = 2, Justify = 3 };
struct Block {
  enum Kind { Heading, Para, Bullet, Number, Code, Table, Rule, PageBreak } kind = Para;
  int level = 0;  // heading 1..4; list depth 0..3
  int num = 0;    // a numbered item's number
  int align = Left;
  std::vector<Run> runs;
  std::vector<std::string> lines;                   // code
  std::vector<std::vector<std::vector<Run>>> rows;  // table: rows of cells of runs
  std::vector<int> col_align;
};
struct Options {
  std::string title, subtitle, author, date;
  std::string page = "A4";
  bool landscape = false, page_numbers = true;
  std::string font = "Calibri", mono = "Consolas";
  double font_size = 11, line_spacing = 1.15;
  double margin_mm[4] = {20, 20, 20, 20};  // top, right, bottom, left
  std::string heading_color = "1F3864", accent_color = "DCE6F1", text_color = "000000";
  std::string header, footer;
};

// ---- small helpers
inline std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
inline std::string lower(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  return s;
}
// "#c00000", "c00000", "#c00", "red" -> "C00000" (empty when not a color)
inline std::string hex_color(std::string v) {
  v = lower(trim(v));
  static const std::pair<const char*, const char*> named[] = {
      {"black", "000000"}, {"white", "FFFFFF"}, {"red", "C00000"}, {"green", "00B050"}, {"blue", "0070C0"},
      {"navy", "1F3864"}, {"gray", "808080"}, {"grey", "808080"}, {"orange", "ED7D31"}, {"yellow", "FFFF00"},
      {"purple", "7030A0"}, {"teal", "008080"}, {"darkred", "8B0000"}, {"darkgreen", "006400"}, {"lightgray", "D9D9D9"}};
  for (const auto& [n, h] : named)
    if (v == n) return h;
  if (!v.empty() && v[0] == '#') v = v.substr(1);
  if (v.size() == 3) v = std::string{v[0], v[0], v[1], v[1], v[2], v[2]};
  if (v.size() != 6) return "";
  for (char c : v)
    if (!isxdigit((unsigned char)c)) return "";
  for (char& c : v) c = (char)toupper((unsigned char)c);
  return v;
}
inline std::string entities(const std::string& s) {
  std::string o;
  for (size_t k = 0; k < s.size(); ++k) {
    if (s[k] == '&') {
      static const std::pair<const char*, const char*> e[] = {{"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""},
                                                             {"&nbsp;", "\xC2\xA0"}, {"&#39;", "'"}};
      bool hit = false;
      for (const auto& [a, b] : e)
        if (s.compare(k, strlen(a), a) == 0) { o += b; k += strlen(a) - 1; hit = true; break; }
      if (hit) continue;
    }
    o += s[k];
  }
  return o;
}

// ---- inline Markdown and HTML into runs
inline void apply_css(Style& st, const std::string& css) {
  size_t a = 0;
  while (a < css.size()) {
    size_t e = css.find(';', a);
    if (e == std::string::npos) e = css.size();
    const std::string d = css.substr(a, e - a);
    const size_t c = d.find(':');
    if (c != std::string::npos) {
      const std::string k = lower(trim(d.substr(0, c))), v = lower(trim(d.substr(c + 1)));
      if (k == "color") st.color = hex_color(v);
      else if (k == "background" || k == "background-color") st.bg = hex_color(v);
      else if (k == "font-size") {
        double n = atof(v.c_str());
        if (v.find("px") != std::string::npos) n *= 0.75;
        if (n >= 5 && n <= 72) st.size = n;
      } else if (k == "font-weight") st.b = v == "bold" || atoi(v.c_str()) >= 600;
      else if (k == "font-style") st.i = v == "italic";
      else if (k == "text-decoration") { st.u = v.find("underline") != std::string::npos; st.s = v.find("line-through") != std::string::npos; }
    }
    a = e + 1;
  }
}
inline std::vector<Run> inlines(const std::string& src) {
  std::vector<Run> out;
  Style cur;
  std::vector<Style> stack;  // (saved styles of open HTML tags)
  std::string text;
  auto flush = [&]() {
    if (!text.empty()) out.push_back(Run{entities(text), cur, false});
    text.clear();
  };
  const std::string& s = src;
  for (size_t k = 0; k < s.size(); ++k) {
    const char c = s[k];
    if (c == '\\' && k + 1 < s.size() && strchr("\\`*_~=<>[]#|", s[k + 1])) { text += s[++k]; continue; }
    if (c == '`') {
      const size_t e = s.find('`', k + 1);
      if (e != std::string::npos) {
        flush();
        Style cs = cur;
        cs.code = true;
        out.push_back(Run{s.substr(k + 1, e - k - 1), cs, false});
        k = e;
        continue;
      }
    }
    if (s.compare(k, 2, "**") == 0) { flush(); cur.b = !cur.b; ++k; continue; }
    if (s.compare(k, 2, "~~") == 0) { flush(); cur.s = !cur.s; ++k; continue; }
    if (s.compare(k, 2, "==") == 0 && (cur.mark || s.find("==", k + 2) != std::string::npos)) { flush(); cur.mark = !cur.mark; ++k; continue; }
    if ((c == '*' || c == '_') && !(c == '_' && k > 0 && isalnum((unsigned char)s[k - 1]))) {
      const bool open = !cur.i && k + 1 < s.size() && s[k + 1] != ' ' && s.find(c, k + 1) != std::string::npos;
      const bool close = cur.i && k > 0 && s[k - 1] != ' ';
      if (open || close) { flush(); cur.i = !cur.i; continue; }
    }
    if (c == '<') {
      const size_t e = s.find('>', k);
      if (e != std::string::npos && e - k < 300) {
        std::string tag = s.substr(k + 1, e - k - 1);
        const bool closing = !tag.empty() && tag[0] == '/';
        if (closing) tag = tag.substr(1);
        const std::string name = lower(tag.substr(0, tag.find_first_of(" /")));
        static const char* known[] = {"b", "strong", "i", "em", "u", "s", "del", "strike", "mark", "span", "br", "sup", "sub", "font"};
        bool is_known = false;
        for (const char* n : known) is_known = is_known || name == n;
        if (is_known) {
          flush();
          if (name == "br") { out.push_back(Run{"", cur, true}); k = e; continue; }
          if (closing) {
            if (!stack.empty()) { cur = stack.back(); stack.pop_back(); }
          } else {
            stack.push_back(cur);
            if (name == "b" || name == "strong") cur.b = true;
            else if (name == "i" || name == "em") cur.i = true;
            else if (name == "u") cur.u = true;
            else if (name == "s" || name == "del" || name == "strike") cur.s = true;
            else if (name == "mark") cur.mark = true;
            else if (name == "span" || name == "font") {
              const size_t st = tag.find("style=");
              if (st != std::string::npos && st + 7 <= tag.size()) {
                const char q = tag[st + 6];
                const size_t qe = tag.find(q, st + 7);
                if (qe != std::string::npos) apply_css(cur, tag.substr(st + 7, qe - st - 7));
              }
              const size_t col = tag.find("color=");
              if (name == "font" && col != std::string::npos && col + 7 <= tag.size()) {
                const char q = tag[col + 6];
                const size_t qe = tag.find(q, col + 7);
                if (qe != std::string::npos) cur.color = hex_color(tag.substr(col + 7, qe - col - 7));
              }
            }
          }
          k = e;
          continue;
        }
      }
    }
    text += c;
  }
  flush();
  return out;
}

// ---- Markdown into blocks
inline std::vector<Block> parse(const std::string& md) {
  std::vector<Block> out;
  std::vector<std::string> lines;
  {
    size_t a = 0;
    while (a <= md.size()) {
      size_t e = md.find('\n', a);
      if (e == std::string::npos) e = md.size();
      std::string l = md.substr(a, e - a);
      if (!l.empty() && l.back() == '\r') l.pop_back();
      lines.push_back(l);
      a = e + 1;
    }
  }
  int align = Left;
  std::string para;
  auto add = [&](Block b) {
    if (b.kind != Block::Table && b.kind != Block::PageBreak && b.kind != Block::Rule) b.align = align;
    out.push_back(std::move(b));
  };
  auto end_para = [&]() {
    if (trim(para).empty()) { para.clear(); return; }
    Block b;
    b.kind = Block::Para;
    b.runs = inlines(trim(para));
    add(b);
    para.clear();
  };
  // a row's cells: \| and a pipe inside a code span stay in the cell ("a \| b", "`ls | sort`": before, they split it;
  // an odd number of backticks: no code spans, or one would take the rest of the row)
  auto cells = [](const std::string& l) {
    std::vector<std::string> c;
    const std::string t = trim(l);
    const bool spans = std::count(t.begin(), t.end(), '`') % 2 == 0;
    std::string cur;
    bool code = false;
    for (size_t k = 0; k < t.size(); ++k) {
      if (t[k] == '\\' && k + 1 < t.size() && t[k + 1] == '|') { cur += '|'; ++k; continue; }
      if (t[k] == '`' && spans) code = !code;
      if (t[k] == '|' && !code) { c.push_back(trim(cur)); cur.clear(); continue; }
      cur += t[k];
    }
    c.push_back(trim(cur));
    if (c.size() > 1 && c.front().empty() && !t.empty() && t[0] == '|') c.erase(c.begin());  // (the leading and trailing pipes)
    if (c.size() > 1 && c.back().empty() && t.size() > 1 && t.back() == '|' && t[t.size() - 2] != '\\') c.pop_back();
    return c;
  };
  for (size_t li = 0; li < lines.size(); ++li) {
    const std::string& l = lines[li];
    const std::string t = trim(l);
    if (t.rfind("```", 0) == 0) {
      end_para();
      Block b;
      b.kind = Block::Code;
      for (++li; li < lines.size() && trim(lines[li]).rfind("```", 0) != 0; ++li) b.lines.push_back(lines[li]);
      add(b);
      continue;
    }
    if (t.rfind(":::", 0) == 0) {  // ::: center / right / justify / left ... :::
      end_para();
      const std::string a = lower(trim(t.substr(3)));
      align = a == "center" ? Center : a == "right" ? Right : a == "justify" ? Justify : Left;
      continue;
    }
    {
      const std::string lt = lower(t);
      if (lt == "\\pagebreak" || lt == "\\newpage" || lt == "<!-- pagebreak -->" || lt == "<!--pagebreak-->" ||
          (lt.find("page-break-") != std::string::npos && lt.rfind("<div", 0) == 0)) {
        end_para();
        Block b;
        b.kind = Block::PageBreak;
        add(b);
        continue;
      }
    }
    if (t.empty()) { end_para(); continue; }
    if (t == "---" || t == "***" || t == "___") {
      end_para();
      Block b;
      b.kind = Block::Rule;
      add(b);
      continue;
    }
    if (t[0] == '#') {
      size_t n = 0;
      while (n < t.size() && t[n] == '#') ++n;
      if (n <= 6 && n < t.size() && t[n] == ' ') {
        end_para();
        Block b;
        b.kind = Block::Heading;
        b.level = (int)std::min<size_t>(n, 4);
        b.runs = inlines(trim(t.substr(n)));
        add(b);
        continue;
      }
    }
    if (t[0] == '|' && li + 1 < lines.size()) {
      const std::string sep = trim(lines[li + 1]);
      if (!sep.empty() && sep[0] == '|' && sep.find("---") != std::string::npos) {  // a table
        end_para();
        Block b;
        b.kind = Block::Table;
        for (const std::string& c : cells(sep)) {
          const bool lft = !c.empty() && c.front() == ':', rgt = !c.empty() && c.back() == ':';
          b.col_align.push_back(lft && rgt ? Center : rgt ? Right : Left);
        }
        auto row = [&](const std::string& x) {
          std::vector<std::vector<Run>> r;
          for (const std::string& c : cells(x)) r.push_back(inlines(c));
          b.rows.push_back(r);
        };
        row(t);
        for (li += 2; li < lines.size() && trim(lines[li]).rfind("|", 0) == 0; ++li) row(lines[li]);
        --li;
        add(b);
        continue;
      }
    }
    {  // list items: "- ", "* ", "+ ", "1. " / "1) " with two spaces per level
      const size_t ind = l.find_first_not_of(' ');
      const std::string r = l.substr(ind);
      size_t d = 0;
      while (d < r.size() && isdigit((unsigned char)r[d])) ++d;
      const bool bullet = r.size() > 2 && (r[0] == '-' || r[0] == '*' || r[0] == '+') && r[1] == ' ';
      const bool number = d > 0 && d + 1 < r.size() && (r[d] == '.' || r[d] == ')') && r[d + 1] == ' ';
      if (bullet || number) {
        end_para();
        Block b;
        b.kind = bullet ? Block::Bullet : Block::Number;
        b.level = (int)std::min<size_t>(ind / 2, 3);
        b.num = number ? atoi(r.c_str()) : 0;
        b.runs = inlines(trim(r.substr(bullet ? 2 : d + 2)));
        add(b);
        continue;
      }
    }
    // a line ending in two spaces or <br>: a line break inside the paragraph
    para += (para.empty() ? "" : (para.size() >= 2 && para.compare(para.size() - 2, 2, "  ") == 0 ? "<br>" : " ")) + t;
    if (l.size() >= 2 && l.compare(l.size() - 2, 2, "  ") == 0) para += "  ";
  }
  end_para();
  return out;
}

// ---- ZIP (stored)
inline uint32_t crc32(const std::string& d) {
  static uint32_t tab[256];
  static bool made = false;
  if (!made) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      tab[i] = c;
    }
    made = true;
  }
  uint32_t c = 0xFFFFFFFFu;
  for (unsigned char x : d) c = tab[(c ^ x) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}
inline std::string zip(const std::vector<std::pair<std::string, std::string>>& files) {
  std::string out, cd;
  auto u16 = [](std::string& s, uint16_t v) { s += (char)(v & 0xFF); s += (char)(v >> 8); };
  auto u32 = [](std::string& s, uint32_t v) { for (int k = 0; k < 4; ++k) s += (char)((v >> (8 * k)) & 0xFF); };
  for (const auto& [name, data] : files) {
    const uint32_t crc = crc32(data), off = (uint32_t)out.size();
    u32(out, 0x04034b50); u16(out, 20); u16(out, 0); u16(out, 0); u16(out, 0); u16(out, 0x21);
    u32(out, crc); u32(out, (uint32_t)data.size()); u32(out, (uint32_t)data.size());
    u16(out, (uint16_t)name.size()); u16(out, 0);
    out += name;
    out += data;
    u32(cd, 0x02014b50); u16(cd, 20); u16(cd, 20); u16(cd, 0); u16(cd, 0); u16(cd, 0); u16(cd, 0x21);
    u32(cd, crc); u32(cd, (uint32_t)data.size()); u32(cd, (uint32_t)data.size());
    u16(cd, (uint16_t)name.size()); u16(cd, 0); u16(cd, 0); u16(cd, 0); u16(cd, 0); u32(cd, 0); u32(cd, off);
    cd += name;
  }
  const uint32_t cdoff = (uint32_t)out.size();
  out += cd;
  u32(out, 0x06054b50); u16(out, 0); u16(out, 0); u16(out, (uint16_t)files.size()); u16(out, (uint16_t)files.size());
  u32(out, (uint32_t)cd.size()); u32(out, cdoff); u16(out, 0);
  return out;
}

// ---- page geometry (points)
inline void page_size(const Options& o, double& w, double& h) {
  const std::string p = lower(o.page);
  if (p == "letter") { w = 612; h = 792; }
  else if (p == "legal") { w = 612; h = 1008; }
  else if (p == "a5") { w = 419.53; h = 595.28; }
  else { w = 595.28; h = 841.89; }
  if (o.landscape) std::swap(w, h);
}
inline double mm_pt(double mm) { return mm * 72.0 / 25.4; }

// (defined with the PDF part below: the fonts' widths)
inline std::vector<double> col_widths(const Block& b, double tw, double size, bool serif, double pad);
inline bool serif_font(const std::string& name);

// ---- DOCX
inline std::string xesc(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c == '&') o += "&amp;";
    else if (c == '<') o += "&lt;";
    else if (c == '>') o += "&gt;";
    else if (c == '"') o += "&quot;";
    else if (c < 0x20 && c != '\t') continue;  // (not allowed in XML 1.0)
    else o += (char)c;
  }
  return o;
}
// w:rPr in the schema's order: rFonts, b, i, strike, color, sz, szCs, highlight, u, shd
inline std::string rpr(const Style& st, const Options& o, bool bold_all = false) {
  std::string r;
  if (st.code) r += "<w:rFonts w:ascii=\"" + xesc(o.mono) + "\" w:hAnsi=\"" + xesc(o.mono) + "\" w:cs=\"" + xesc(o.mono) + "\"/>";
  if (st.b || bold_all) r += "<w:b/>";
  if (st.i) r += "<w:i/>";
  if (st.s) r += "<w:strike/>";
  if (!st.color.empty()) r += "<w:color w:val=\"" + st.color + "\"/>";
  if (st.size > 0) {
    const std::string hp = std::to_string((int)std::lround(st.size * 2));
    r += "<w:sz w:val=\"" + hp + "\"/><w:szCs w:val=\"" + hp + "\"/>";
  }
  if (st.mark && st.bg.empty()) r += "<w:highlight w:val=\"yellow\"/>";
  if (st.u) r += "<w:u w:val=\"single\"/>";
  if (!st.bg.empty()) r += "<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"" + st.bg + "\"/>";
  else if (st.code) r += "<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"F1F1EE\"/>";
  return r.empty() ? "" : "<w:rPr>" + r + "</w:rPr>";
}
inline std::string docx_runs(const std::vector<Run>& runs, const Options& o, bool bold_all = false) {
  std::string out;
  for (const Run& r : runs) {
    if (r.br) { out += "<w:r><w:br/></w:r>"; continue; }
    out += "<w:r>" + rpr(r.st, o, bold_all) + "<w:t xml:space=\"preserve\">" + xesc(r.text) + "</w:t></w:r>";
  }
  return out;
}
inline const char* jc(int a) { return a == Center ? "center" : a == Right ? "right" : a == Justify ? "both" : "left"; }

inline std::string docx(const std::string& md, const Options& o) {
  double pw, ph;
  page_size(o, pw, ph);
  const int tw = (int)std::lround(pw * 20), th = (int)std::lround(ph * 20);
  int mg[4];
  for (int k = 0; k < 4; ++k) mg[k] = (int)std::lround(mm_pt(o.margin_mm[k]) * 20);
  const int text_w = tw - mg[1] - mg[3];
  std::string body;
  // a paragraph; ppr: its properties after pStyle, in the schema's order (numPr, pBdr, shd, spacing, ind, jc)
  auto para = [&](const std::string& style, const std::string& runs, const std::string& ppr = "") {
    body += "<w:p><w:pPr>" + (style.empty() ? "" : "<w:pStyle w:val=\"" + style + "\"/>") + ppr + "</w:pPr>" + runs + "</w:p>";
  };
  auto jcx = [](int a) { return a == Left ? std::string() : "<w:jc w:val=\"" + std::string(jc(a)) + "\"/>"; };
  if (!trim(o.title).empty()) para("Title", docx_runs({Run{o.title}}, o));
  if (!trim(o.subtitle).empty()) para("Subtitle", docx_runs({Run{o.subtitle}}, o));
  if (!trim(o.author).empty() || !trim(o.date).empty()) {
    Style m;
    m.color = "595959";
    para("", docx_runs({Run{trim(o.author) + (o.author.empty() || o.date.empty() ? "" : " \xC2\xB7 ") + trim(o.date), m}}, o));
  }
  int lists = 0;
  bool in_num = false;
  for (const Block& b : parse(md)) {
    if (b.kind != Block::Number) in_num = false;
    switch (b.kind) {
      case Block::Heading: para("Heading" + std::to_string(b.level), docx_runs(b.runs, o), jcx(b.align)); break;
      case Block::Para: para("", docx_runs(b.runs, o), jcx(b.align)); break;
      case Block::Bullet:
        para("ListParagraph", docx_runs(b.runs, o),
             "<w:numPr><w:ilvl w:val=\"" + std::to_string(b.level) + "\"/><w:numId w:val=\"1\"/></w:numPr>" + jcx(b.align));
        break;
      case Block::Number:
        if (!in_num) { ++lists; in_num = true; }
        para("ListParagraph", docx_runs(b.runs, o),
             "<w:numPr><w:ilvl w:val=\"" + std::to_string(b.level) + "\"/><w:numId w:val=\"" + std::to_string(1 + lists) + "\"/></w:numPr>" + jcx(b.align));
        break;
      case Block::Code:
        for (const std::string& l : b.lines) para("Code", "<w:r><w:t xml:space=\"preserve\">" + xesc(l) + "</w:t></w:r>");
        para("", "");
        break;
      case Block::Rule: para("", "", "<w:pBdr><w:bottom w:val=\"single\" w:sz=\"6\" w:space=\"1\" w:color=\"A6A6A6\"/></w:pBdr>"); break;
      case Block::PageBreak: body += "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"; break;
      case Block::Table: {
        size_t cols = 0;
        for (const auto& r : b.rows) cols = std::max(cols, r.size());
        if (!cols) break;
        std::vector<int> cw;
        int total = 0;
        for (double w : col_widths(b, text_w / 20.0, o.font_size, serif_font(o.font), 6)) { cw.push_back((int)std::lround(w * 20)); total += cw.back(); }
        const std::string bd = "w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"A6A6A6\"";
        body += "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"" + std::to_string(total) + "\" w:type=\"dxa\"/>"
                "<w:tblBorders><w:top " + bd + "/><w:left " + bd + "/><w:bottom " + bd + "/><w:right " + bd + "/><w:insideH " + bd +
                "/><w:insideV " + bd + "/></w:tblBorders><w:tblLayout w:type=\"fixed\"/><w:tblLook w:val=\"04A0\" w:firstRow=\"1\" "
                "w:lastRow=\"0\" w:firstColumn=\"0\" w:lastColumn=\"0\" w:noHBand=\"0\" w:noVBand=\"1\"/></w:tblPr><w:tblGrid>";
        for (size_t c = 0; c < cols; ++c) body += "<w:gridCol w:w=\"" + std::to_string(cw[c]) + "\"/>";
        body += "</w:tblGrid>";
        for (size_t r = 0; r < b.rows.size(); ++r) {
          body += r == 0 ? "<w:tr><w:trPr><w:tblHeader/></w:trPr>" : "<w:tr>";
          for (size_t c = 0; c < cols; ++c) {
            const std::vector<Run> cell = c < b.rows[r].size() ? b.rows[r][c] : std::vector<Run>{};
            const int a = c < b.col_align.size() ? b.col_align[c] : Left;
            body += "<w:tc><w:tcPr><w:tcW w:w=\"" + std::to_string(cw[c]) + "\" w:type=\"dxa\"/>" +
                    (r == 0 ? "<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"" + o.accent_color + "\"/>" : std::string()) +
                    "</w:tcPr><w:p><w:pPr><w:spacing w:before=\"40\" w:after=\"40\"/>" + jcx(a) + "</w:pPr>" + docx_runs(cell, o, r == 0) + "</w:p></w:tc>";
          }
          body += "</w:tr>";
        }
        body += "</w:tbl>";
        para("", "");
        break;
      }
    }
  }
  if (body.empty()) para("", "");
  const bool hdr = !trim(o.header).empty(), ftr = !trim(o.footer).empty() || o.page_numbers;
  const std::string W = "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
                        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"";
  std::string sect = "<w:sectPr>";
  if (hdr) sect += "<w:headerReference w:type=\"default\" r:id=\"rId3\"/>";
  if (ftr) sect += "<w:footerReference w:type=\"default\" r:id=\"rId4\"/>";
  sect += "<w:pgSz w:w=\"" + std::to_string(tw) + "\" w:h=\"" + std::to_string(th) + "\"" + (o.landscape ? " w:orient=\"landscape\"" : "") +
          "/><w:pgMar w:top=\"" + std::to_string(mg[0]) + "\" w:right=\"" + std::to_string(mg[1]) + "\" w:bottom=\"" + std::to_string(mg[2]) +
          "\" w:left=\"" + std::to_string(mg[3]) + "\" w:header=\"567\" w:footer=\"567\" w:gutter=\"0\"/></w:sectPr>";
  const std::string document = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:document " + W + "><w:body>" + body + sect +
                               "</w:body></w:document>";
  const std::string font = xesc(o.font), hc = o.heading_color;
  const int sz = (int)std::lround(o.font_size * 2), line = (int)std::lround(240 * std::clamp(o.line_spacing, 0.8, 3.0));
  auto style = [](const std::string& id, const std::string& name, const std::string& ppr, const std::string& rp) {
    return "<w:style w:type=\"paragraph\" w:styleId=\"" + id + "\"><w:name w:val=\"" + name + "\"/><w:basedOn w:val=\"Normal\"/>"
           "<w:next w:val=\"Normal\"/><w:qFormat/>" + (ppr.empty() ? "" : "<w:pPr>" + ppr + "</w:pPr>") + (rp.empty() ? "" : "<w:rPr>" + rp + "</w:rPr>") + "</w:style>";
  };
  auto hsz = [&](double k) { const std::string v = std::to_string((int)std::lround(o.font_size * 2 * k)); return "<w:sz w:val=\"" + v + "\"/><w:szCs w:val=\"" + v + "\"/>"; };
  const std::string mono = xesc(o.mono);
  const std::string styles =
      "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:styles " + W + ">"
      "<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii=\"" + font + "\" w:eastAsia=\"" + font + "\" w:hAnsi=\"" + font + "\" w:cs=\"" + font + "\"/>"
      "<w:color w:val=\"" + o.text_color + "\"/><w:sz w:val=\"" + std::to_string(sz) + "\"/><w:szCs w:val=\"" + std::to_string(sz) + "\"/>"
      "<w:lang w:val=\"en-US\" w:eastAsia=\"en-US\" w:bidi=\"ar-SA\"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>"
      "<w:spacing w:after=\"120\" w:line=\"" + std::to_string(line) + "\" w:lineRule=\"auto\"/></w:pPr></w:pPrDefault></w:docDefaults>"
      "<w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\"><w:name w:val=\"Normal\"/><w:qFormat/></w:style>"
      "<w:style w:type=\"character\" w:default=\"1\" w:styleId=\"DefaultParagraphFont\"><w:name w:val=\"Default Paragraph Font\"/><w:uiPriority w:val=\"1\"/><w:semiHidden/><w:unhideWhenUsed/></w:style>" +
      style("Title", "Title", "<w:spacing w:after=\"80\"/>", "<w:b/><w:color w:val=\"" + hc + "\"/>" + hsz(2.4)) +
      style("Subtitle", "Subtitle", "<w:spacing w:after=\"160\"/>", "<w:color w:val=\"595959\"/>" + hsz(1.3)) +
      style("Heading1", "heading 1", "<w:keepNext/><w:keepLines/><w:spacing w:before=\"320\" w:after=\"120\"/><w:outlineLvl w:val=\"0\"/>",
            "<w:b/><w:color w:val=\"" + hc + "\"/>" + hsz(1.6)) +
      style("Heading2", "heading 2", "<w:keepNext/><w:keepLines/><w:spacing w:before=\"240\" w:after=\"80\"/><w:outlineLvl w:val=\"1\"/>",
            "<w:b/><w:color w:val=\"" + hc + "\"/>" + hsz(1.3)) +
      style("Heading3", "heading 3", "<w:keepNext/><w:keepLines/><w:spacing w:before=\"200\" w:after=\"60\"/><w:outlineLvl w:val=\"2\"/>",
            "<w:b/><w:color w:val=\"" + hc + "\"/>" + hsz(1.12)) +
      style("Heading4", "heading 4", "<w:keepNext/><w:keepLines/><w:spacing w:before=\"160\" w:after=\"40\"/><w:outlineLvl w:val=\"3\"/>",
            "<w:b/><w:i/><w:color w:val=\"" + hc + "\"/>" + hsz(1.0)) +
      style("ListParagraph", "List Paragraph", "<w:spacing w:after=\"40\"/><w:ind w:left=\"720\"/><w:contextualSpacing/>", "") +
      style("Code", "Code", "<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"F4F4F2\"/><w:spacing w:after=\"0\" w:line=\"240\" w:lineRule=\"auto\"/>",
            "<w:rFonts w:ascii=\"" + mono + "\" w:hAnsi=\"" + mono + "\" w:cs=\"" + mono + "\"/>" + hsz(0.85)) +
      style("Header", "header", "<w:jc w:val=\"right\"/>", "<w:color w:val=\"7F7F7F\"/>" + hsz(0.8)) +
      style("Footer", "footer", "<w:jc w:val=\"center\"/>", "<w:color w:val=\"7F7F7F\"/>" + hsz(0.8)) +
      "<w:style w:type=\"table\" w:default=\"1\" w:styleId=\"TableNormal\"><w:name w:val=\"Normal Table\"/><w:uiPriority w:val=\"99\"/><w:semiHidden/>"
      "<w:unhideWhenUsed/><w:tblPr><w:tblInd w:w=\"0\" w:type=\"dxa\"/><w:tblCellMar><w:top w:w=\"0\" w:type=\"dxa\"/><w:left w:w=\"108\" w:type=\"dxa\"/>"
      "<w:bottom w:w=\"0\" w:type=\"dxa\"/><w:right w:w=\"108\" w:type=\"dxa\"/></w:tblCellMar></w:tblPr></w:style>"
      "<w:style w:type=\"table\" w:styleId=\"TableGrid\"><w:name w:val=\"Table Grid\"/><w:basedOn w:val=\"TableNormal\"/><w:uiPriority w:val=\"39\"/>"
      "<w:pPr><w:spacing w:after=\"0\" w:line=\"240\" w:lineRule=\"auto\"/></w:pPr></w:style></w:styles>";
  std::string numbering = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:numbering " + W + ">";
  auto abstract = [](int id, bool bullet) {
    std::string a = "<w:abstractNum w:abstractNumId=\"" + std::to_string(id) + "\"><w:multiLevelType w:val=\"hybridMultilevel\"/>";
    const char* bul[4] = {"\xE2\x80\xA2", "o", "\xE2\x96\xAA", "\xE2\x80\xA2"};
    const char* fmt[4] = {"decimal", "lowerLetter", "lowerRoman", "decimal"};
    for (int l = 0; l < 4; ++l)
      a += "<w:lvl w:ilvl=\"" + std::to_string(l) + "\"><w:start w:val=\"1\"/><w:numFmt w:val=\"" + (bullet ? std::string("bullet") : fmt[l]) +
           "\"/><w:lvlText w:val=\"" + (bullet ? std::string(bul[l]) : "%" + std::to_string(l + 1) + ".") + "\"/><w:lvlJc w:val=\"left\"/>"
           "<w:pPr><w:ind w:left=\"" + std::to_string(720 + 360 * l) + "\" w:hanging=\"360\"/></w:pPr>" +
           (bullet && l == 1 ? "<w:rPr><w:rFonts w:ascii=\"Courier New\" w:hAnsi=\"Courier New\" w:cs=\"Courier New\"/></w:rPr>" : "") + "</w:lvl>";
    return a + "</w:abstractNum>";
  };
  numbering += abstract(0, true) + abstract(1, false) + "<w:num w:numId=\"1\"><w:abstractNumId w:val=\"0\"/></w:num>";
  for (int k = 1; k <= lists; ++k)  // (each numbered list starts again at 1)
    numbering += "<w:num w:numId=\"" + std::to_string(1 + k) + "\"><w:abstractNumId w:val=\"1\"/><w:lvlOverride w:ilvl=\"0\"><w:startOverride w:val=\"1\"/></w:lvlOverride></w:num>";
  numbering += "</w:numbering>";
  const std::string R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
  std::vector<std::pair<std::string, std::string>> parts;
  std::string types = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
      "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
      "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
      "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
      "<Override PartName=\"/word/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/>"
      "<Override PartName=\"/word/numbering.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml\"/>"
      "<Override PartName=\"/word/settings.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml\"/>";
  if (hdr) types += "<Override PartName=\"/word/header1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml\"/>";
  if (ftr) types += "<Override PartName=\"/word/footer1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml\"/>";
  types += "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
           "<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/></Types>";
  std::string drels = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
      "<Relationship Id=\"rId1\" Type=\"" + R + "/styles\" Target=\"styles.xml\"/>"
      "<Relationship Id=\"rId2\" Type=\"" + R + "/numbering\" Target=\"numbering.xml\"/>"
      "<Relationship Id=\"rId5\" Type=\"" + R + "/settings\" Target=\"settings.xml\"/>";
  if (hdr) drels += "<Relationship Id=\"rId3\" Type=\"" + R + "/header\" Target=\"header1.xml\"/>";
  if (ftr) drels += "<Relationship Id=\"rId4\" Type=\"" + R + "/footer\" Target=\"footer1.xml\"/>";
  drels += "</Relationships>";
  parts.push_back({"[Content_Types].xml", types});
  parts.push_back({"_rels/.rels",
      "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
      "<Relationship Id=\"rId1\" Type=\"" + R + "/officeDocument\" Target=\"word/document.xml\"/>"
      "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>"
      "<Relationship Id=\"rId3\" Type=\"" + R + "/extended-properties\" Target=\"docProps/app.xml\"/></Relationships>"});
  parts.push_back({"docProps/core.xml",
      "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
      "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:dcterms=\"http://purl.org/dc/terms/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
      "<dc:title>" + xesc(o.title) + "</dc:title><dc:creator>" + xesc(o.author.empty() ? std::string("arcinferzero") : o.author) + "</dc:creator></cp:coreProperties>"});
  parts.push_back({"docProps/app.xml",
      "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\">"
      "<Application>arcinferzero</Application></Properties>"});
  parts.push_back({"word/_rels/document.xml.rels", drels});
  parts.push_back({"word/document.xml", document});
  parts.push_back({"word/styles.xml", styles});
  parts.push_back({"word/numbering.xml", numbering});
  parts.push_back({"word/settings.xml",
      "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:settings " + W + "><w:defaultTabStop w:val=\"720\"/>"
      "<w:characterSpacingControl w:val=\"doNotCompress\"/><w:compat><w:compatSetting w:name=\"compatibilityMode\" "
      "w:uri=\"http://schemas.microsoft.com/office/word\" w:val=\"15\"/></w:compat></w:settings>"});
  if (hdr)
    parts.push_back({"word/header1.xml", "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:hdr " + W + "><w:p><w:pPr><w:pStyle w:val=\"Header\"/></w:pPr>" +
                                             docx_runs(inlines(o.header), o) + "</w:p></w:hdr>"});
  if (ftr) {
    std::string f = "<w:p><w:pPr><w:pStyle w:val=\"Footer\"/></w:pPr>" + docx_runs(inlines(o.footer), o);
    if (o.page_numbers)
      f += std::string(trim(o.footer).empty() ? "" : "<w:r><w:t xml:space=\"preserve\">    </w:t></w:r>") +
           "<w:r><w:fldChar w:fldCharType=\"begin\"/></w:r><w:r><w:instrText xml:space=\"preserve\"> PAGE </w:instrText></w:r>"
           "<w:r><w:fldChar w:fldCharType=\"separate\"/></w:r><w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType=\"end\"/></w:r>"
           "<w:r><w:t xml:space=\"preserve\"> / </w:t></w:r>"
           "<w:r><w:fldChar w:fldCharType=\"begin\"/></w:r><w:r><w:instrText xml:space=\"preserve\"> NUMPAGES </w:instrText></w:r>"
           "<w:r><w:fldChar w:fldCharType=\"separate\"/></w:r><w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType=\"end\"/></w:r>";
    parts.push_back({"word/footer1.xml", "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:ftr " + W + ">" + f + "</w:p></w:ftr>"});
  }
  return zip(parts);
}

// ---- RTF (also written as .doc: Word, LibreOffice and WordPad open RTF content under that name as a normal document)
// A style sheet (Normal, heading 1-4, Title, Subtitle) so headings stay headings (navigation pane, contents), the fonts
// and colors in tables, direct run formatting, lists as indented paragraphs with their bullet or number, tables with
// borders, a repeated and shaded header row and the same column widths as the DOCX, header / footer with PAGE /
// NUMPAGES fields; text as \uN escapes outside ASCII.
inline std::string rtf_text(const std::string& s) {
  std::string o;
  for (size_t k = 0; k < s.size();) {
    const unsigned char c = (unsigned char)s[k];
    uint32_t cp = c;
    int n = 1;
    if (c >= 0xF0 && k + 3 < s.size()) { cp = ((c & 7u) << 18) | (((unsigned char)s[k + 1] & 63u) << 12) | (((unsigned char)s[k + 2] & 63u) << 6) | ((unsigned char)s[k + 3] & 63u); n = 4; }
    else if (c >= 0xE0 && k + 2 < s.size()) { cp = ((c & 15u) << 12) | (((unsigned char)s[k + 1] & 63u) << 6) | ((unsigned char)s[k + 2] & 63u); n = 3; }
    else if (c >= 0xC0 && k + 1 < s.size()) { cp = ((c & 31u) << 6) | ((unsigned char)s[k + 1] & 63u); n = 2; }
    else if (c >= 0x80) cp = '?';
    k += n;
    if (cp == '\\' || cp == '{' || cp == '}') { o += '\\'; o += (char)cp; }
    else if (cp == '\t') o += "\\tab ";
    else if (cp < 0x20) continue;
    else if (cp < 0x80) o += (char)cp;
    else if (cp < 0x10000) o += "\\u" + std::to_string((int16_t)cp) + "?";
    else {  // (outside the BMP: a UTF-16 surrogate pair)
      const uint32_t v = cp - 0x10000;
      o += "\\u" + std::to_string((int16_t)(0xD800 + (v >> 10))) + "?\\u" + std::to_string((int16_t)(0xDC00 + (v & 0x3FF))) + "?";
    }
  }
  return o;
}
inline std::string rtf(const std::string& md, const Options& o) {
  double pw, ph;
  page_size(o, pw, ph);
  int mg[4];
  for (int k = 0; k < 4; ++k) mg[k] = (int)std::lround(mm_pt(o.margin_mm[k]) * 20);
  const int text_w = (int)std::lround(pw * 20) - mg[1] - mg[3];
  const std::vector<Block> blocks = parse(md);
  // colors: 1 text, 2 heading, 3 accent (table header), 4 gray, 5 code background, 6 yellow, then the runs' own
  std::vector<std::string> colors = {o.text_color, o.heading_color, o.accent_color, "595959", "F4F4F2", "FFFF00"};
  auto cidx = [&](const std::string& hex) {
    if (hex.size() != 6) return 0;
    for (size_t k = 0; k < colors.size(); ++k)
      if (colors[k] == hex) return (int)k + 1;
    colors.push_back(hex);
    return (int)colors.size();
  };
  const int fs = (int)std::lround(o.font_size * 2), sl = (int)std::lround(240 * std::clamp(o.line_spacing, 0.8, 3.0));
  auto runs = [&](const std::vector<Run>& rr, bool bold_all = false) {
    std::string s;
    for (const Run& r : rr) {
      if (r.br) { s += "\\line "; continue; }
      std::string f;
      if (r.st.code) f += "\\f1\\fs" + std::to_string((int)std::lround(o.font_size * 1.7)) + "\\chcbpat5";
      if (r.st.b || bold_all) f += "\\b";
      if (r.st.i) f += "\\i";
      if (r.st.u) f += "\\ul";
      if (r.st.s) f += "\\strike";
      if (!r.st.color.empty()) f += "\\cf" + std::to_string(cidx(r.st.color));
      if (r.st.size > 0) f += "\\fs" + std::to_string((int)std::lround(r.st.size * 2));
      if (!r.st.bg.empty()) f += "\\chcbpat" + std::to_string(cidx(r.st.bg));
      else if (r.st.mark) f += "\\highlight6";
      s += f.empty() ? rtf_text(r.text) : "{" + f + " " + rtf_text(r.text) + "}";
    }
    return s;
  };
  auto qa = [](int a) { return a == Center ? "\\qc" : a == Right ? "\\qr" : a == Justify ? "\\qj" : "\\ql"; };
  const std::string normal = "\\s0\\f0\\fs" + std::to_string(fs) + "\\cf1\\sa120\\sl" + std::to_string(sl) + "\\slmult1";
  auto hstyle = [&](int level) {
    const double k[5] = {0, 1.6, 1.3, 1.12, 1.0};
    const int before[5] = {0, 320, 240, 200, 160};
    return "\\s" + std::to_string(level) + "\\keepn\\sb" + std::to_string(before[level]) + "\\sa" + std::to_string(level <= 2 ? 120 : 60) +
           "\\outlinelevel" + std::to_string(level - 1) + "\\b" + std::string(level == 4 ? "\\i" : "") + "\\cf2\\fs" +
           std::to_string((int)std::lround(o.font_size * 2 * k[level]));
  };
  std::string body;
  if (!trim(o.title).empty()) body += "\\pard\\plain\\s5\\sa80\\b\\cf2\\fs" + std::to_string((int)std::lround(o.font_size * 4.8)) + " " + rtf_text(o.title) + "\\par\n";
  if (!trim(o.subtitle).empty()) body += "\\pard\\plain\\s6\\sa160\\cf4\\fs" + std::to_string((int)std::lround(o.font_size * 2.6)) + " " + rtf_text(o.subtitle) + "\\par\n";
  if (!trim(o.author).empty() || !trim(o.date).empty())
    body += "\\pard\\plain" + normal + "\\cf4 " + rtf_text(trim(o.author) + (o.author.empty() || o.date.empty() ? "" : " \xC2\xB7 ") + trim(o.date)) + "\\par\n";
  int last_num = 0;
  for (const Block& b : blocks) {
    switch (b.kind) {
      case Block::Heading: body += "\\pard\\plain" + hstyle(b.level) + qa(b.align) + " " + runs(b.runs) + "\\par\n"; break;
      case Block::Para: body += "\\pard\\plain" + normal + qa(b.align) + " " + runs(b.runs) + "\\par\n"; break;
      case Block::Bullet:
      case Block::Number: {
        std::string mark;
        if (b.kind == Block::Bullet) mark = b.level % 2 ? "o" : "\xE2\x80\xA2";
        else { last_num = b.num ? b.num : last_num + 1; mark = std::to_string(last_num) + "."; }
        const int li = 720 + 360 * b.level;
        body += "\\pard\\plain" + normal + "\\sa40\\fi-360\\li" + std::to_string(li) + "\\tx" + std::to_string(li) + qa(b.align) + " " +
                rtf_text(mark) + "\\tab " + runs(b.runs) + "\\par\n";
        break;
      }
      case Block::Code:
        for (const std::string& l : b.lines)
          body += "\\pard\\plain\\s0\\f1\\fs" + std::to_string((int)std::lround(o.font_size * 1.7)) + "\\cf1\\sa0\\sl240\\slmult1\\cbpat5 " + rtf_text(l) + "\\par\n";
        body += "\\pard\\plain" + normal + "\\par\n";
        break;
      case Block::Rule: body += "\\pard\\plain" + normal + "\\brdrb\\brdrs\\brdrw10\\brsp20\\brdrcf4 \\par\n"; break;
      case Block::PageBreak: body += "\\pard\\plain" + normal + "\\page\\par\n"; break;
      case Block::Table: {
        size_t cols = 0;
        for (const auto& r : b.rows) cols = std::max(cols, r.size());
        if (!cols) break;
        std::vector<int> cx;
        int x = 0;
        for (double w : col_widths(b, text_w / 20.0, o.font_size, serif_font(o.font), 6)) { x += (int)std::lround(w * 20); cx.push_back(x); }
        for (size_t r = 0; r < b.rows.size(); ++r) {
          std::string row = "\\trowd\\trgaph108\\trleft0" + std::string(r == 0 ? "\\trhdr" : "");
          for (size_t c = 0; c < cols; ++c)
            row += "\\clbrdrt\\brdrs\\brdrw10\\brdrcf4\\clbrdrl\\brdrs\\brdrw10\\brdrcf4\\clbrdrb\\brdrs\\brdrw10\\brdrcf4\\clbrdrr\\brdrs\\brdrw10\\brdrcf4" +
                   std::string(r == 0 ? "\\clcbpat3" : "") + "\\cellx" + std::to_string(cx[c]);
          body += row + "\n";
          for (size_t c = 0; c < cols; ++c) {
            const std::vector<Run> cell = c < b.rows[r].size() ? b.rows[r][c] : std::vector<Run>{};
            const int a = c < b.col_align.size() ? b.col_align[c] : Left;
            body += "\\pard\\plain\\intbl\\s0\\f0\\fs" + std::to_string(fs) + "\\cf1\\sb40\\sa40" + qa(a) + " " + runs(cell, r == 0) + "\\cell\n";
          }
          body += row + "\\row\n";
        }
        body += "\\pard\\plain" + normal + "\\par\n";
        break;
      }
    }
    if (b.kind != Block::Number) last_num = 0;
  }
  // (built after the body: the runs may add colors)
  std::string head = "{\\rtf1\\ansi\\ansicpg1252\\deff0\\uc1{\\fonttbl{\\f0\\f" + std::string(serif_font(o.font) ? "roman" : "swiss") +
                     "\\fcharset0 " + rtf_text(o.font) + ";}{\\f1\\fmodern\\fcharset0 " + rtf_text(o.mono) + ";}}\n{\\colortbl;";
  for (const std::string& c : colors)
    head += "\\red" + std::to_string(strtol(c.substr(0, 2).c_str(), nullptr, 16)) + "\\green" + std::to_string(strtol(c.substr(2, 2).c_str(), nullptr, 16)) +
            "\\blue" + std::to_string(strtol(c.substr(4, 2).c_str(), nullptr, 16)) + ";";
  head += "}\n{\\stylesheet{" + normal + " Normal;}";
  for (int l = 1; l <= 4; ++l) head += "{" + hstyle(l) + "\\sbasedon0\\snext0 heading " + std::to_string(l) + ";}";
  head += "{\\s5\\sa80\\b\\cf2\\fs" + std::to_string((int)std::lround(o.font_size * 4.8)) + "\\sbasedon0\\snext0 Title;}"
          "{\\s6\\sa160\\cf4\\fs" + std::to_string((int)std::lround(o.font_size * 2.6)) + "\\sbasedon0\\snext0 Subtitle;}}\n";
  head += "{\\info{\\title " + rtf_text(o.title) + "}{\\author " + rtf_text(o.author.empty() ? std::string("arcinferzero") : o.author) + "}}\n";
  head += "\\paperw" + std::to_string((int)std::lround(pw * 20)) + "\\paperh" + std::to_string((int)std::lround(ph * 20)) + "\\margt" +
          std::to_string(mg[0]) + "\\margr" + std::to_string(mg[1]) + "\\margb" + std::to_string(mg[2]) + "\\margl" + std::to_string(mg[3]) +
          (o.landscape ? "\\landscape" : "") + "\\headery567\\footery567\n\\sectd\n";
  if (!trim(o.header).empty())
    head += "{\\header\\pard\\plain\\qr\\f0\\fs" + std::to_string((int)std::lround(o.font_size * 1.6)) + "\\cf4 " + runs(inlines(o.header)) + "\\par}\n";
  if (!trim(o.footer).empty() || o.page_numbers) {
    head += "{\\footer\\pard\\plain\\qc\\f0\\fs" + std::to_string((int)std::lround(o.font_size * 1.6)) + "\\cf4 " + runs(inlines(o.footer));
    if (o.page_numbers)
      head += std::string(trim(o.footer).empty() ? "" : "    ") + "{\\field{\\*\\fldinst PAGE}{\\fldrslt 1}} / {\\field{\\*\\fldinst NUMPAGES}{\\fldrslt 1}}";
    head += "\\par}\n";
  }
  return head + body + "}";
}

// ---- PDF
// character widths (1/1000 em) of WinAnsi 32..126, from the fonts' AFM metrics: Helvetica, Helvetica-Bold, Times-Roman,
// Times-Bold (the obliques / italics are taken as the upright ones)
inline const int* widths(bool serif, bool bold) {
  static const int hr[95] = {278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556, 556, 556, 556,
                             556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
                             667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556, 333, 556, 556, 500, 556, 556, 278, 556,
                             556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584};
  static const int hb[95] = {278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556, 556, 556, 556,
                             556, 556, 333, 333, 584, 584, 584, 611, 975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
                             667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556, 333, 556, 611, 556, 611, 556, 333, 611,
                             611, 278, 278, 556, 278, 889, 611, 611, 611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584};
  static const int tr[95] = {250, 333, 408, 500, 500, 833, 778, 180, 333, 333, 500, 564, 250, 333, 250, 278, 500, 500, 500, 500, 500, 500, 500, 500,
                             500, 500, 278, 278, 564, 564, 564, 444, 921, 722, 667, 667, 722, 611, 556, 722, 722, 333, 389, 722, 611, 889, 722, 722,
                             556, 722, 667, 556, 611, 722, 722, 944, 722, 722, 611, 333, 278, 333, 469, 500, 333, 444, 500, 444, 500, 444, 333, 500,
                             500, 278, 278, 500, 278, 778, 500, 500, 500, 500, 333, 389, 278, 500, 500, 722, 500, 500, 444, 480, 200, 480, 541};
  static const int tb[95] = {250, 333, 555, 500, 500, 1000, 833, 278, 333, 333, 500, 570, 250, 333, 250, 278, 500, 500, 500, 500, 500, 500, 500, 500,
                             500, 500, 333, 333, 570, 570, 570, 500, 930, 722, 667, 722, 722, 667, 611, 778, 778, 389, 500, 778, 667, 944, 722, 778,
                             611, 778, 722, 556, 667, 722, 722, 1000, 722, 722, 667, 333, 278, 333, 581, 500, 333, 500, 556, 444, 556, 444, 333, 500,
                             556, 278, 333, 556, 278, 833, 556, 500, 556, 556, 444, 389, 333, 556, 500, 722, 500, 500, 444, 394, 220, 394, 520};
  return serif ? (bold ? tb : tr) : (bold ? hb : hr);
}
// UTF-8 -> WinAnsi bytes (Latin-1, plus the curly quotes, dashes, bullet, ellipsis, euro, trademark); others '?'
inline std::string winansi(const std::string& s) {
  std::string o;
  for (size_t k = 0; k < s.size();) {
    const unsigned char c = (unsigned char)s[k];
    uint32_t cp = c;
    int n = 1;
    if (c >= 0xF0 && k + 3 < s.size()) { cp = ((c & 7u) << 18) | (((unsigned char)s[k + 1] & 63u) << 12) | (((unsigned char)s[k + 2] & 63u) << 6) | ((unsigned char)s[k + 3] & 63u); n = 4; }
    else if (c >= 0xE0 && k + 2 < s.size()) { cp = ((c & 15u) << 12) | (((unsigned char)s[k + 1] & 63u) << 6) | ((unsigned char)s[k + 2] & 63u); n = 3; }
    else if (c >= 0xC0 && k + 1 < s.size()) { cp = ((c & 31u) << 6) | ((unsigned char)s[k + 1] & 63u); n = 2; }
    else if (c >= 0x80) cp = '?';
    k += n;
    switch (cp) {
      case 0x2018: o += '\x91'; continue;
      case 0x2019: o += '\x92'; continue;
      case 0x201C: o += '\x93'; continue;
      case 0x201D: o += '\x94'; continue;
      case 0x2022: o += '\x95'; continue;
      case 0x2013: o += '\x96'; continue;
      case 0x2014: o += '\x97'; continue;
      case 0x2026: o += '\x85'; continue;
      case 0x20AC: o += '\x80'; continue;
      case 0x2122: o += '\x99'; continue;
      case 0x00A0: o += ' '; continue;
      case '\t': o += "    "; continue;
      default: break;
    }
    o += (cp >= 32 && cp < 127) || (cp >= 0xA1 && cp <= 0xFF) ? (char)cp : '?';
  }
  return o;
}

// the width of one WinAnsi character (1/1000 em)
inline double glyph_w(unsigned char c, bool serif, bool bold, bool mono) {
  if (mono) return 600;
  if (c >= 32 && c < 127) return widths(serif, bold)[c - 32];
  if (c == 0xB7) return serif ? 250 : 278;  // (middle dot)
  if (c == 0x95) return 350;
  if (c == 0x96) return 500;
  if (c == 0x97 || c == 0x85) return 1000;
  if (c == 0x91 || c == 0x92) return serif ? 333 : 222;
  if (c == 0x93 || c == 0x94) return serif ? 444 : 333;
  return serif ? 480 : 556;  // (accented letters: about a lower-case letter)
}
inline double str_w(const std::string& winansi_text, bool serif, bool bold, double size) {
  double w = 0;
  for (unsigned char c : winansi_text) w += glyph_w(c, serif, bold, false);
  return w * size / 1000.0;
}
inline bool serif_font(const std::string& name) {
  const std::string f = lower(name);
  if (f.find("sans") != std::string::npos) return false;
  return f.find("times") != std::string::npos || f.find("georgia") != std::string::npos || f.find("cambria") != std::string::npos ||
         f.find("garamond") != std::string::npos || f.find("serif") != std::string::npos || f.find("book") != std::string::npos;
}
// a table's column widths (points) in the text width tw: by their content (the widest cell as one line), each at least
// its longest word, shrunk proportionally when the content is wider than the page
inline std::vector<double> col_widths(const Block& b, double tw, double size, bool serif, double pad) {
  size_t cols = 0;
  for (const auto& r : b.rows) cols = std::max(cols, r.size());
  std::vector<double> nat(cols, 2 * pad + 10), mn(cols, 2 * pad + 10);
  for (size_t r = 0; r < b.rows.size(); ++r)
    for (size_t c = 0; c < b.rows[r].size(); ++c) {
      std::string t;
      for (const Run& x : b.rows[r][c]) t += x.text + (x.br ? " " : "");
      t = winansi(t);
      nat[c] = std::max(nat[c], str_w(t, serif, r == 0, size) + 2 * pad + 2);
      size_t a = 0;
      while (a <= t.size()) {
        size_t e = t.find(' ', a);
        if (e == std::string::npos) e = t.size();
        mn[c] = std::max(mn[c], std::min(tw / 2, str_w(t.substr(a, e - a), serif, r == 0, size) + 2 * pad + 2));
        a = e + 1;
      }
    }
  double sn = 0, sm = 0;
  for (size_t c = 0; c < cols; ++c) { sn += nat[c]; sm += mn[c]; }
  std::vector<double> w(cols);
  if (sn <= tw) {
    for (size_t c = 0; c < cols; ++c) w[c] = nat[c] * tw / sn;
  } else if (sm >= tw) {
    for (size_t c = 0; c < cols; ++c) w[c] = mn[c] * tw / sm;
  } else {
    const double extra = tw - sm, want = sn - sm;
    for (size_t c = 0; c < cols; ++c) w[c] = mn[c] + (want > 0 ? extra * (nat[c] - mn[c]) / want : extra / cols);
  }
  return w;
}

struct Pdf {
  const Options& o;
  bool serif;
  double W = 0, H = 0, L = 0, R = 0, T = 0, B = 0;
  std::vector<std::string> pages;
  std::string cur;
  double y = 0;
  explicit Pdf(const Options& opt) : o(opt) {
    page_size(o, W, H);
    T = mm_pt(o.margin_mm[0]); R = mm_pt(o.margin_mm[1]); B = mm_pt(o.margin_mm[2]); L = mm_pt(o.margin_mm[3]);
    serif = serif_font(o.font);
    y = H - T;
  }
  // fonts: F1-F4 Helvetica (regular, bold, oblique, bold oblique), F5 / F6 Courier (regular, bold), F7-F10 Times
  int font_of(const Style& st, bool bold_all) const {
    const bool b = st.b || bold_all;
    if (st.code) return b ? 6 : 5;
    const int base = serif ? 7 : 1;
    return base + (b ? 1 : 0) + (st.i ? 2 : 0);
  }
  double char_w(unsigned char c, int font) const {
    return glyph_w(c, font >= 7, font == 2 || font == 4 || font == 6 || font == 8 || font == 10, font == 5 || font == 6);
  }
  double text_w(const std::string& t, int font, double size) const {
    double w = 0;
    for (unsigned char c : t) w += char_w(c, font);
    return w * size / 1000.0;
  }
  static std::string pesc(const std::string& s) {
    std::string out;
    for (char c : s) {
      if (c == '(' || c == ')' || c == '\\') out += '\\';
      out += c;
    }
    return out;
  }
  static std::string rgb(const std::string& hex, const char* op) {
    if (hex.size() != 6) return "";
    char b[48];
    snprintf(b, sizeof b, "%.3f %.3f %.3f %s ", strtol(hex.substr(0, 2).c_str(), nullptr, 16) / 255.0,
             strtol(hex.substr(2, 2).c_str(), nullptr, 16) / 255.0, strtol(hex.substr(4, 2).c_str(), nullptr, 16) / 255.0, op);
    return b;
  }
  void new_page() { pages.push_back(cur); cur.clear(); y = H - T; }
  bool need(double h) {  // (true: a new page began)
    if (y - h < B && y < H - T - 1) { new_page(); return true; }
    return false;
  }
  void text_at(double x, double yy, int font, double size, const std::string& t, const std::string& color = "") {
    char b[96];
    snprintf(b, sizeof b, "BT /F%d %.2f Tf %.2f %.2f Td ", font, size, x, yy);
    cur += (color.empty() ? "" : rgb(color, "rg")) + b + "(" + pesc(t) + ") Tj ET" + (color.empty() ? "" : " 0 g") + "\n";
  }

  // ---- a paragraph's runs laid out in lines, each line placed (aligned) and drawn
  struct Seg {
    std::string t;  // (WinAnsi, no spaces inside except a trailing one in sp)
    int font;
    double size, w;
    Style st;
    bool sp;  // a space follows
  };
  void flow(const std::vector<Run>& runs, double x0, double width, double size, int align, const std::string& color = "",
            bool bold_all = false, double first_indent = 0) {
    std::vector<std::vector<Seg>> lines(1);
    std::vector<double> lw(1, 0);
    double x = first_indent;
    auto room = [&]() { return width - x; };
    for (const Run& r : runs) {
      if (r.br) { lines.emplace_back(); lw.push_back(0); x = 0; continue; }
      const double sz = r.st.size > 0 ? r.st.size : size;
      const int font = font_of(r.st, bold_all);
      const std::string t = winansi(r.text);
      size_t a = 0;
      while (a < t.size()) {
        size_t e = t.find(' ', a);
        const bool sp = e != std::string::npos;
        if (!sp) e = t.size();
        std::string word = t.substr(a, e - a);
        a = e + (sp ? 1 : 0);
        double ww = text_w(word, font, sz);
        const double spw = sp ? text_w(" ", font, sz) : 0;
        if (word.empty()) {  // (a run of spaces: one space at most)
          if (!lines.back().empty() && !lines.back().back().sp) { lines.back().back().sp = true; x += spw; }
          continue;
        }
        if (ww > room() && !lines.back().empty()) { lines.emplace_back(); lw.push_back(0); x = 0; }
        while (ww > width && word.size() > 1) {  // (a word wider than the line: cut by characters)
          size_t n = word.size();
          while (n > 1 && text_w(word.substr(0, n), font, sz) > width - x) --n;
          lines.back().push_back(Seg{word.substr(0, n), font, sz, text_w(word.substr(0, n), font, sz), r.st, false});
          lines.emplace_back();
          lw.push_back(0);
          x = 0;
          word = word.substr(n);
          ww = text_w(word, font, sz);
        }
        lines.back().push_back(Seg{word, font, sz, ww, r.st, sp});
        x += ww + spw;
      }
    }
    const double ls = std::clamp(o.line_spacing, 0.8, 3.0);
    for (size_t li = 0; li < lines.size(); ++li) {
      auto& ln = lines[li];
      double maxsz = size, w = 0;
      int spaces = 0;
      for (size_t k = 0; k < ln.size(); ++k) {
        maxsz = std::max(maxsz, ln[k].size);
        w += ln[k].w + (ln[k].sp && k + 1 < ln.size() ? text_w(" ", ln[k].font, ln[k].size) : 0);
        if (ln[k].sp && k + 1 < ln.size()) ++spaces;
      }
      const double lead = maxsz * 1.2 * ls;
      need(lead);
      const double base = y - maxsz * 0.95;
      const double indent = li == 0 ? first_indent : 0;
      double sx = x0 + indent, extra = 0;
      const double avail = width - indent;
      if (align == Center) sx += (avail - w) / 2;
      else if (align == Right) sx += avail - w;
      else if (align == Justify && li + 1 < lines.size() && spaces > 0) extra = (avail - w) / spaces;
      for (size_t k = 0; k < ln.size(); ++k) {
        const Seg& g = ln[k];
        const double spw = g.sp && k + 1 < ln.size() ? text_w(" ", g.font, g.size) + extra : 0;
        const std::string bg = !g.st.bg.empty() ? g.st.bg : g.st.mark ? "FFFF00" : g.st.code ? "EFEFEC" : "";
        if (!bg.empty()) {  // (behind the following space too when the next piece has the same background)
          const bool joined = k + 1 < ln.size() && ln[k + 1].st.bg == g.st.bg && ln[k + 1].st.mark == g.st.mark && ln[k + 1].st.code == g.st.code;
          char rc[96];
          snprintf(rc, sizeof rc, "%.2f %.2f %.2f %.2f re f 0 g\n", sx - 0.5, base - g.size * 0.22, g.w + (joined ? spw : 0) + 1, g.size * 1.12);
          cur += rgb(bg, "rg") + rc;
        }
        const std::string col = !g.st.color.empty() ? g.st.color : color;
        text_at(sx, base, g.font, g.size, g.t, col);
        if (g.st.u || g.st.s) {
          char ln2[128];
          const double yy = g.st.u ? base - g.size * 0.12 : base + g.size * 0.3;
          snprintf(ln2, sizeof ln2, "%.2f w %.2f %.2f m %.2f %.2f l S\n", std::max(0.4, g.size * 0.05), sx, yy, sx + g.w, yy);
          cur += (col.empty() ? "" : rgb(col, "RG")) + ln2 + (col.empty() ? "" : "0 G\n");
        }
        sx += g.w + spw;
      }
      y -= lead;
    }
  }

  std::string finish() {
    pages.push_back(cur);
    std::vector<std::string> obj;
    const int n = (int)pages.size();
    const int first_page = 13;  // (objects 1-12: catalog, pages, 10 fonts; then info; then per page: page, content)
    std::string kids;
    for (int p = 0; p < n; ++p) kids += std::to_string(first_page + 1 + 2 * p) + " 0 R ";
    obj.push_back("<< /Type /Catalog /Pages 2 0 R >>");
    obj.push_back("<< /Type /Pages /Kids [" + kids + "] /Count " + std::to_string(n) + " >>");
    for (const char* f : {"Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique", "Courier", "Courier-Bold",
                          "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"})
      obj.push_back(std::string("<< /Type /Font /Subtype /Type1 /BaseFont /") + f + " /Encoding /WinAnsiEncoding >>");
    obj.push_back("<< /Title (" + pesc(winansi(o.title)) + ") /Author (" + pesc(winansi(o.author)) + ") /Producer (arcinferzero) >>");
    std::string fonts;
    for (int f = 1; f <= 10; ++f) fonts += "/F" + std::to_string(f) + " " + std::to_string(2 + f) + " 0 R ";
    const int hf = serif ? 7 : 1;
    for (int p = 0; p < n; ++p) {
      std::string c = pages[p];
      const double fs = std::max(7.0, o.font_size * 0.8);
      if (!trim(o.header).empty()) {
        const std::string t = winansi(o.header);
        c += "0.5 g BT /F" + std::to_string(hf) + " " + std::to_string(fs) + " Tf " + std::to_string(W - R - text_w(t, hf, fs)) + " " +
             std::to_string(H - T / 2) + " Td (" + pesc(t) + ") Tj ET 0 g\n";
      }
      std::string foot = winansi(o.footer);
      if (o.page_numbers) foot += (foot.empty() ? "" : "    ") + std::to_string(p + 1) + " / " + std::to_string(n);
      if (!foot.empty())
        c += "0.5 g BT /F" + std::to_string(hf) + " " + std::to_string(fs) + " Tf " + std::to_string((W - text_w(foot, hf, fs)) / 2) + " " +
             std::to_string(B / 2) + " Td (" + pesc(foot) + ") Tj ET 0 g\n";
      char pg[384];
      snprintf(pg, sizeof pg, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] /Resources << /Font << %s>> >> /Contents %d 0 R >>", W, H,
               fonts.c_str(), first_page + 2 + 2 * p);
      obj.push_back(pg);
      obj.push_back("<< /Length " + std::to_string(c.size()) + " >>\nstream\n" + c + "endstream");
    }
    std::string out = "%PDF-1.4\n%\xE2\xE3\xCF\xD3\n";
    std::vector<size_t> off;
    for (size_t k = 0; k < obj.size(); ++k) {
      off.push_back(out.size());
      out += std::to_string(k + 1) + " 0 obj\n" + obj[k] + "\nendobj\n";
    }
    const size_t xref = out.size();
    out += "xref\n0 " + std::to_string(obj.size() + 1) + "\n0000000000 65535 f \n";
    for (size_t x : off) {
      char b[24];
      snprintf(b, sizeof b, "%010zu 00000 n \n", x);
      out += b;
    }
    out += "trailer\n<< /Size " + std::to_string(obj.size() + 1) + " /Root 1 0 R /Info 13 0 R >>\nstartxref\n" + std::to_string(xref) + "\n%%EOF\n";
    return out;
  }
};

inline std::string pdf(const std::string& md, const Options& o) {
  Pdf p(o);
  const double tw = p.W - p.L - p.R, fs = o.font_size, ls = std::clamp(o.line_spacing, 0.8, 3.0);
  if (!trim(o.title).empty()) { p.flow(inlines(o.title), p.L, tw, fs * 2.2, Left, o.heading_color, true); p.y -= fs * 0.3; }
  if (!trim(o.subtitle).empty()) { p.flow(inlines(o.subtitle), p.L, tw, fs * 1.3, Left, "595959"); p.y -= fs * 0.3; }
  if (!trim(o.author).empty() || !trim(o.date).empty()) {
    p.flow({Run{trim(o.author) + (o.author.empty() || o.date.empty() ? "" : " \xC2\xB7 ") + trim(o.date)}}, p.L, tw, fs * 0.95, Left, "595959");
    p.y -= fs * 0.6;
  }
  int last_num = 0;
  for (const Block& b : parse(md)) {
    switch (b.kind) {
      case Block::Heading: {
        const double k[5] = {0, 1.6, 1.3, 1.12, 1.0}, sz = fs * k[b.level];
        p.y -= sz * (b.level <= 2 ? 0.8 : 0.5);
        p.need(sz * 1.6 + fs * 1.2 * ls * 4);  // (a heading stays with what follows: about three lines or a table row)
        std::vector<Run> rr = b.runs;
        if (b.level == 4) for (Run& r : rr) r.st.i = true;
        p.flow(rr, p.L, tw, sz, b.align, o.heading_color, true);
        p.y -= sz * 0.25;
        break;
      }
      case Block::Para: p.flow(b.runs, p.L, tw, fs, b.align); p.y -= fs * 0.5; break;
      case Block::Bullet:
      case Block::Number: {
        const double ind = fs * 1.6 + fs * 1.4 * b.level, x = p.L + ind;
        std::string mark;
        if (b.kind == Block::Bullet) mark = b.level % 2 ? "-" : "\x95";
        else { last_num = b.num ? b.num : last_num + 1; mark = std::to_string(last_num) + "."; }
        p.need(fs * 1.2 * ls);
        const int mf = p.serif ? 7 : 1;
        p.text_at(x - p.text_w(mark, mf, fs) - fs * 0.45, p.y - fs * 0.95, mf, fs, mark);
        p.flow(b.runs, x, tw - ind, fs, b.align);
        p.y -= fs * 0.15;
        break;
      }
      case Block::Code: {
        const double cs = fs * 0.85, lead = cs * 1.3, cw = 600 * cs / 1000.0;
        const size_t maxc = std::max<size_t>(10, (size_t)((tw - 8) / cw));
        p.y -= fs * 0.2;
        for (const std::string& l0 : b.lines) {
          std::string t = winansi(l0);
          do {  // (a long line wrapped, not cut)
            const std::string part = t.substr(0, maxc);
            t = t.size() > maxc ? t.substr(maxc) : "";
            p.need(lead);
            char rect[112];
            snprintf(rect, sizeof rect, "0.957 0.957 0.949 rg %.2f %.2f %.2f %.2f re f 0 g\n", p.L - 4, p.y - lead, tw + 8, lead);
            p.cur += rect;
            p.text_at(p.L, p.y - cs * 1.0, 5, cs, part);
            p.y -= lead;
          } while (!t.empty());
        }
        p.y -= fs * 0.6;
        break;
      }
      case Block::Rule: {
        p.need(fs);
        char ln[112];
        snprintf(ln, sizeof ln, "0.65 G 0.6 w %.2f %.2f m %.2f %.2f l S 0 G\n", p.L, p.y - fs * 0.5, p.W - p.R, p.y - fs * 0.5);
        p.cur += ln;
        p.y -= fs;
        break;
      }
      case Block::PageBreak: p.new_page(); break;
      case Block::Table: {
        size_t cols = 0;
        for (const auto& r : b.rows) cols = std::max(cols, r.size());
        if (!cols) break;
        const double size = fs * 0.92, pad = 4;
        const std::vector<double> cwv = col_widths(b, tw, size, p.serif, pad);
        std::vector<double> cx(cols + 1, p.L);
        for (size_t c = 0; c < cols; ++c) cx[c + 1] = cx[c] + cwv[c];
        // one row: each cell laid out on a scratch page to find its height, then drawn at y
        auto draw_row = [&](size_t r) {
          double h = 0;
          std::vector<std::string> laid(cols);
          for (size_t c = 0; c < cols; ++c) {
            Pdf q(o);
            q.serif = p.serif;
            q.y = 100000;  // (no page breaks while measuring)
            q.B = -1e9;
            const std::vector<Run> cell = c < b.rows[r].size() ? b.rows[r][c] : std::vector<Run>{};
            const int a = c < b.col_align.size() ? b.col_align[c] : Left;
            q.flow(cell, cx[c] + pad, cwv[c] - 2 * pad, size, a, "", r == 0);
            h = std::max(h, 100000 - q.y);
            laid[c] = q.cur;
          }
          h += 2 * pad;
          const bool broke = p.need(h);
          if (broke && r > 0) return false;  // (the header row first on the new page: the caller draws it)
          // the cells' drawing moved from y = 100000 down to this row's top
          const double dy = (p.y - pad) - 100000;
          if (r == 0) {
            char box[128];
            snprintf(box, sizeof box, "%.2f %.2f %.2f %.2f re f 0 g\n", p.L, p.y - h, tw, h);
            p.cur += Pdf::rgb(o.accent_color, "rg") + box;
          }
          for (size_t c = 0; c < cols; ++c) {
            p.cur += "q 1 0 0 1 0 " + std::to_string(dy) + " cm\n" + laid[c] + "Q\n";
            char box[112];
            snprintf(box, sizeof box, "0.6 G 0.5 w %.2f %.2f %.2f %.2f re S 0 G\n", cx[c], p.y - h, cwv[c], h);
            p.cur += box;
          }
          p.y -= h;
          return true;
        };
        for (size_t r = 0; r < b.rows.size(); ++r)
          if (!draw_row(r)) { draw_row(0); draw_row(r); }  // (a page break inside the table: its header row again)
        p.y -= fs * 0.6;
        break;
      }
    }
    if (b.kind != Block::Number) last_num = 0;
  }
  return p.finish();
}

}  // namespace doc
