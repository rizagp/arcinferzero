// Documents the model writes (the web UI's create_document tool): Markdown in, a .docx or .pdf file out, written here
// with no library. The Markdown subset: # .. #### headings, paragraphs, **bold**, *italic*, `code`, - / * / 1. lists
// (two spaces per level), ``` code blocks, | tables |, --- rules.
//   DOCX: a ZIP (stored, CRC-32) of WordprocessingML: styles (Title, Heading1-4, Code), numbering (bullets, numbers),
//         tables with borders; opens in Word, LibreOffice, Google Docs.
//   PDF:  PDF 1.4, A4, the standard fonts (Helvetica, -Bold, -Oblique, -BoldOblique, Courier) in WinAnsi encoding
//         (Latin-1 plus the usual quotes and dashes; other characters become '?'), lines wrapped by the fonts' character
//         widths, page breaks, page numbers.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace doc {

// ---- Markdown into blocks of runs
struct Run {
  std::string text;  // (UTF-8)
  bool b = false, i = false, code = false;
};
struct Block {
  enum Kind { Heading, Para, Bullet, Number, Code, Table, Rule } kind = Para;
  int level = 0;  // heading 1..4; list depth 0..
  int num = 0;    // a numbered item's number
  std::vector<Run> runs;
  std::vector<std::string> lines;                 // code
  std::vector<std::vector<std::vector<Run>>> rows;  // table: rows of cells of runs
};

inline std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
// **bold**, *italic* / _italic_, `code`
inline std::vector<Run> inlines(const std::string& s) {
  std::vector<Run> out;
  Run cur;
  bool b = false, it = false;
  auto flush = [&]() {
    if (!cur.text.empty()) out.push_back(cur);
    cur = Run{};
    cur.b = b;
    cur.i = it;
  };
  for (size_t k = 0; k < s.size(); ++k) {
    if (s[k] == '`') {
      const size_t e = s.find('`', k + 1);
      if (e != std::string::npos) {
        flush();
        Run c;
        c.text = s.substr(k + 1, e - k - 1);
        c.code = true;
        out.push_back(c);
        k = e;
        continue;
      }
    }
    if (s.compare(k, 2, "**") == 0) { flush(); b = !b; cur.b = b; ++k; continue; }
    if ((s[k] == '*' || s[k] == '_') && (k + 1 < s.size() && s[k + 1] != ' ') && (k == 0 || !isalnum((unsigned char)s[k - 1]))) {
      if (s.find(s[k], k + 1) != std::string::npos || it) { flush(); it = !it; cur.i = it; continue; }
    }
    if (it && (s[k] == '*' || s[k] == '_') && (k + 1 == s.size() || !isalnum((unsigned char)s[k + 1]))) { flush(); it = false; cur.i = false; continue; }
    cur.text += s[k];
  }
  flush();
  return out;
}
inline std::vector<Block> parse(const std::string& md) {
  std::vector<Block> out;
  std::vector<std::string> lines;
  {
    size_t a = 0;
    while (a <= md.size()) {
      size_t e = md.find('\n', a);
      if (e == std::string::npos) e = md.size();
      lines.push_back(md.substr(a, e - a));
      a = e + 1;
    }
  }
  std::string para;
  auto end_para = [&]() {
    if (trim(para).empty()) { para.clear(); return; }
    Block b;
    b.kind = Block::Para;
    b.runs = inlines(trim(para));
    out.push_back(b);
    para.clear();
  };
  auto cells = [](const std::string& l) {
    std::vector<std::string> c;
    std::string t = trim(l);
    if (!t.empty() && t[0] == '|') t = t.substr(1);
    if (!t.empty() && t.back() == '|') t.pop_back();
    size_t a = 0;
    while (true) {
      const size_t e = t.find('|', a);
      c.push_back(trim(t.substr(a, e == std::string::npos ? std::string::npos : e - a)));
      if (e == std::string::npos) break;
      a = e + 1;
    }
    return c;
  };
  for (size_t li = 0; li < lines.size(); ++li) {
    const std::string& raw = lines[li];
    std::string l = raw;
    if (!l.empty() && l.back() == '\r') l.pop_back();
    const std::string t = trim(l);
    if (t.rfind("```", 0) == 0) {  // code block
      end_para();
      Block b;
      b.kind = Block::Code;
      for (++li; li < lines.size() && trim(lines[li]).rfind("```", 0) != 0; ++li) {
        std::string c = lines[li];
        if (!c.empty() && c.back() == '\r') c.pop_back();
        b.lines.push_back(c);
      }
      out.push_back(b);
      continue;
    }
    if (t.empty()) { end_para(); continue; }
    if (t == "---" || t == "***" || t == "___") { end_para(); Block b; b.kind = Block::Rule; out.push_back(b); continue; }
    if (t[0] == '#') {
      size_t n = 0;
      while (n < t.size() && t[n] == '#') ++n;
      if (n <= 6 && n < t.size() && t[n] == ' ') {
        end_para();
        Block b;
        b.kind = Block::Heading;
        b.level = (int)std::min<size_t>(n, 4);
        b.runs = inlines(trim(t.substr(n)));
        out.push_back(b);
        continue;
      }
    }
    if (t[0] == '|' && li + 1 < lines.size() && trim(lines[li + 1]).find("---") != std::string::npos && trim(lines[li + 1])[0] == '|') {
      end_para();  // table: header, the --- line, rows
      Block b;
      b.kind = Block::Table;
      auto row = [&](const std::string& x) {
        std::vector<std::vector<Run>> r;
        for (const std::string& c : cells(x)) r.push_back(inlines(c));
        b.rows.push_back(r);
      };
      row(t);
      for (li += 2; li < lines.size() && trim(lines[li]).rfind("|", 0) == 0; ++li) row(lines[li]);
      --li;
      out.push_back(b);
      continue;
    }
    {  // list items: "- ", "* ", "+ ", "1. " with two spaces per level
      size_t ind = l.find_first_not_of(' ');
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
        out.push_back(b);
        continue;
      }
    }
    para += (para.empty() ? "" : " ") + t;
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

// ---- DOCX
inline std::string xesc(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c == '&') o += "&amp;";
    else if (c == '<') o += "&lt;";
    else if (c == '>') o += "&gt;";
    else if (c == '"') o += "&quot;";
    else if (c < 0x20 && c != '\t' && c != '\n') continue;  // (not allowed in XML 1.0)
    else o += (char)c;
  }
  return o;
}
inline std::string docx_runs(const std::vector<Run>& runs) {
  std::string o;
  for (const Run& r : runs) {
    o += "<w:r>";
    if (r.b || r.i || r.code) {
      o += "<w:rPr>";
      if (r.code) o += "<w:rFonts w:ascii=\"Consolas\" w:hAnsi=\"Consolas\" w:cs=\"Consolas\"/><w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"F1F1EE\"/>";
      if (r.b) o += "<w:b/>";
      if (r.i) o += "<w:i/>";
      o += "</w:rPr>";
    }
    o += "<w:t xml:space=\"preserve\">" + xesc(r.text) + "</w:t></w:r>";
  }
  return o;
}
inline std::string docx(const std::string& title, const std::string& md) {
  std::string body;
  auto para = [&](const std::string& style, const std::string& runs, const std::string& extra = "") {
    body += "<w:p><w:pPr>" + (style.empty() ? "" : "<w:pStyle w:val=\"" + style + "\"/>") + extra + "</w:pPr>" + runs + "</w:p>";
  };
  if (!trim(title).empty()) para("Title", docx_runs({Run{title}}));
  int list_id = 0, last = -1;  // (numbered lists: a new numbering instance each time a list starts)
  for (const Block& b : parse(md)) {
    if (b.kind != Block::Number) last = -1;
    switch (b.kind) {
      case Block::Heading: para("Heading" + std::to_string(b.level), docx_runs(b.runs)); break;
      case Block::Para: para("", docx_runs(b.runs)); break;
      case Block::Bullet:
        para("ListParagraph", docx_runs(b.runs), "<w:numPr><w:ilvl w:val=\"" + std::to_string(b.level) + "\"/><w:numId w:val=\"1\"/></w:numPr>");
        break;
      case Block::Number:
        if (last < 0) ++list_id;
        last = 1;
        para("ListParagraph", docx_runs(b.runs),
             "<w:numPr><w:ilvl w:val=\"" + std::to_string(b.level) + "\"/><w:numId w:val=\"" + std::to_string(1 + list_id) + "\"/></w:numPr>");
        break;
      case Block::Code:
        for (const std::string& l : b.lines) para("Code", "<w:r><w:t xml:space=\"preserve\">" + xesc(l) + "</w:t></w:r>");
        break;
      case Block::Rule:
        para("", "", "<w:pBdr><w:bottom w:val=\"single\" w:sz=\"6\" w:space=\"1\" w:color=\"999999\"/></w:pBdr>");
        break;
      case Block::Table: {
        body += "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"5000\" w:type=\"pct\"/><w:tblBorders>"
                "<w:top w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/><w:left w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/>"
                "<w:bottom w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/><w:right w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/>"
                "<w:insideH w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/><w:insideV w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/>"
                "</w:tblBorders></w:tblPr>";
        for (size_t r = 0; r < b.rows.size(); ++r) {
          body += "<w:tr>";
          for (const auto& cell : b.rows[r]) {
            std::vector<Run> cr = cell;
            if (r == 0) for (Run& x : cr) x.b = true;  // (the header row)
            body += "<w:tc><w:tcPr>" + std::string(r == 0 ? "<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"EDEDED\"/>" : "") +
                    "</w:tcPr><w:p>" + docx_runs(cr) + "</w:p></w:tc>";
          }
          body += "</w:tr>";
        }
        body += "</w:tbl><w:p/>";
        break;
      }
    }
  }
  const std::string W = "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"";
  const std::string document = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:document " + W + "><w:body>" + body +
                               "<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/><w:pgMar w:top=\"1247\" w:right=\"1134\" w:bottom=\"1247\" "
                               "w:left=\"1134\" w:header=\"709\" w:footer=\"709\" w:gutter=\"0\"/></w:sectPr></w:body></w:document>";
  auto style = [](const std::string& id, const std::string& name, const std::string& ppr, const std::string& rpr) {
    return "<w:style w:type=\"paragraph\" w:styleId=\"" + id + "\"><w:name w:val=\"" + name + "\"/><w:basedOn w:val=\"Normal\"/>"
           "<w:next w:val=\"Normal\"/><w:qFormat/><w:pPr>" + ppr + "</w:pPr><w:rPr>" + rpr + "</w:rPr></w:style>";
  };
  const std::string styles =
      "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:styles " + W + ">"
      "<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii=\"Calibri\" w:hAnsi=\"Calibri\" w:cs=\"Calibri\" w:eastAsia=\"Calibri\"/>"
      "<w:sz w:val=\"22\"/><w:szCs w:val=\"22\"/><w:lang w:val=\"en-US\"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>"
      "<w:spacing w:after=\"120\" w:line=\"264\" w:lineRule=\"auto\"/></w:pPr></w:pPrDefault></w:docDefaults>"
      "<w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\"><w:name w:val=\"Normal\"/><w:qFormat/></w:style>" +
      style("Title", "Title", "<w:spacing w:after=\"240\"/>", "<w:b/><w:sz w:val=\"40\"/><w:color w:val=\"1F3864\"/>") +
      style("Heading1", "heading 1", "<w:keepNext/><w:spacing w:before=\"280\" w:after=\"100\"/><w:outlineLvl w:val=\"0\"/>", "<w:b/><w:sz w:val=\"32\"/><w:color w:val=\"1F3864\"/>") +
      style("Heading2", "heading 2", "<w:keepNext/><w:spacing w:before=\"220\" w:after=\"80\"/><w:outlineLvl w:val=\"1\"/>", "<w:b/><w:sz w:val=\"28\"/><w:color w:val=\"2E5496\"/>") +
      style("Heading3", "heading 3", "<w:keepNext/><w:spacing w:before=\"180\" w:after=\"60\"/><w:outlineLvl w:val=\"2\"/>", "<w:b/><w:sz w:val=\"24\"/>") +
      style("Heading4", "heading 4", "<w:keepNext/><w:spacing w:before=\"160\" w:after=\"40\"/><w:outlineLvl w:val=\"3\"/>", "<w:b/><w:i/><w:sz w:val=\"22\"/>") +
      style("ListParagraph", "List Paragraph", "<w:spacing w:after=\"40\"/><w:ind w:left=\"720\"/>", "") +
      style("Code", "Code", "<w:spacing w:after=\"0\" w:line=\"240\" w:lineRule=\"auto\"/><w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"F4F4F2\"/>",
            "<w:rFonts w:ascii=\"Consolas\" w:hAnsi=\"Consolas\" w:cs=\"Consolas\"/><w:sz w:val=\"19\"/>") +
      "<w:style w:type=\"table\" w:styleId=\"TableGrid\"><w:name w:val=\"Table Grid\"/><w:tblPr><w:tblCellMar><w:left w:w=\"100\" w:type=\"dxa\"/>"
      "<w:right w:w=\"100\" w:type=\"dxa\"/></w:tblCellMar></w:tblPr></w:style></w:styles>";
  std::string numbering = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:numbering " + W + ">";
  auto abstract = [](int id, bool bullet) {
    std::string a = "<w:abstractNum w:abstractNumId=\"" + std::to_string(id) + "\"><w:multiLevelType w:val=\"hybridMultilevel\"/>";
    const char* bul[4] = {"\xE2\x80\xA2", "\xE2\x97\xA6", "\xE2\x96\xAA", "\xE2\x80\xA2"};
    const char* fmt[4] = {"decimal", "lowerLetter", "lowerRoman", "decimal"};
    for (int l = 0; l < 4; ++l)
      a += "<w:lvl w:ilvl=\"" + std::to_string(l) + "\"><w:start w:val=\"1\"/><w:numFmt w:val=\"" + (bullet ? std::string("bullet") : fmt[l]) +
           "\"/><w:lvlText w:val=\"" + (bullet ? std::string(bul[l]) : "%" + std::to_string(l + 1) + ".") + "\"/><w:lvlJc w:val=\"left\"/>"
           "<w:pPr><w:ind w:left=\"" + std::to_string(720 + 360 * l) + "\" w:hanging=\"360\"/></w:pPr></w:lvl>";
    return a + "</w:abstractNum>";
  };
  numbering += abstract(0, true) + abstract(1, false) + "<w:num w:numId=\"1\"><w:abstractNumId w:val=\"0\"/></w:num>";
  for (int k = 1; k <= list_id; ++k)  // (each numbered list starts again at 1)
    numbering += "<w:num w:numId=\"" + std::to_string(1 + k) + "\"><w:abstractNumId w:val=\"1\"/><w:lvlOverride w:ilvl=\"0\"><w:startOverride w:val=\"1\"/></w:lvlOverride></w:num>";
  numbering += "</w:numbering>";
  const std::string R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
  return zip({
      {"[Content_Types].xml",
       "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
       "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
       "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
       "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
       "<Override PartName=\"/word/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/>"
       "<Override PartName=\"/word/numbering.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml\"/>"
       "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/></Types>"},
      {"_rels/.rels",
       "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
       "<Relationship Id=\"rId1\" Type=\"" + R + "/officeDocument\" Target=\"word/document.xml\"/>"
       "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/></Relationships>"},
      {"docProps/core.xml",
       "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
       "xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title>" + xesc(title) + "</dc:title><dc:creator>arcinferzero</dc:creator></cp:coreProperties>"},
      {"word/_rels/document.xml.rels",
       "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
       "<Relationship Id=\"rId1\" Type=\"" + R + "/styles\" Target=\"styles.xml\"/>"
       "<Relationship Id=\"rId2\" Type=\"" + R + "/numbering\" Target=\"numbering.xml\"/></Relationships>"},
      {"word/document.xml", document},
      {"word/styles.xml", styles},
      {"word/numbering.xml", numbering},
  });
}

// ---- PDF
// widths of the WinAnsi characters 32..126 in 1/1000 em (the fonts' AFM metrics); other characters: approximations
inline const int* helv_widths(bool bold) {
  static const int reg[95] = {278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556,
                              556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778,
                              722, 278, 500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278,
                              278, 278, 469, 556, 333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
                              556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584};
  static const int bld[95] = {278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556,
                              556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611, 975, 722, 722, 722, 722, 667, 611, 778,
                              722, 278, 556, 722, 611, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333,
                              278, 333, 584, 556, 333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,
                              611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584};
  return bold ? bld : reg;
}
// UTF-8 -> WinAnsi bytes (Latin-1, plus the curly quotes, dashes, bullet, ellipsis, euro); others '?'
inline std::string winansi(const std::string& s) {
  std::string o;
  for (size_t k = 0; k < s.size();) {
    unsigned char c = (unsigned char)s[k];
    uint32_t cp = c;
    int n = 1;
    if (c >= 0xF0 && k + 3 < s.size() + 0) { cp = ((c & 7u) << 18) | ((s[k + 1] & 63u) << 12) | ((s[k + 2] & 63u) << 6) | (s[k + 3] & 63u); n = 4; }
    else if (c >= 0xE0 && k + 2 < s.size()) { cp = ((c & 15u) << 12) | ((s[k + 1] & 63u) << 6) | (s[k + 2] & 63u); n = 3; }
    else if (c >= 0xC0 && k + 1 < s.size()) { cp = ((c & 31u) << 6) | (s[k + 1] & 63u); n = 2; }
    k += n;
    switch (cp) {
      case 0x2018: o += '\x91'; continue; case 0x2019: o += '\x92'; continue; case 0x201C: o += '\x93'; continue;
      case 0x201D: o += '\x94'; continue; case 0x2022: o += '\x95'; continue; case 0x2013: o += '\x96'; continue;
      case 0x2014: o += '\x97'; continue; case 0x2026: o += '\x85'; continue; case 0x20AC: o += '\x80'; continue;
      case 0x2122: o += '\x99'; continue; case 0x00A0: o += ' '; continue;
      default: break;
    }
    if (cp == '\t') { o += "    "; continue; }
    o += (cp >= 32 && cp < 127) || (cp >= 0xA1 && cp <= 0xFF) ? (char)cp : '?';
  }
  return o;
}
inline double char_w(unsigned char c, bool bold, bool mono) {
  if (mono) return 600;
  if (c >= 32 && c < 127) return helv_widths(bold)[c - 32];
  if (c == 0x95) return 350;
  if (c == 0x96) return 556;
  if (c == 0x97 || c == 0x85) return 1000;
  return 556;  // (accented letters: about a lower-case letter)
}

struct Pdf {
  struct Seg {  // a piece of a line in one font
    std::string text;  // (WinAnsi)
    int font;          // 1 Helvetica, 2 Bold, 3 Oblique, 4 BoldOblique, 5 Courier
  };
  std::vector<std::string> pages;  // content streams
  std::string cur;
  double y = 0;
  const double W = 595.28, H = 841.89, L = 56.7, R = 56.7, T = 62, B = 62;
  void page() {
    if (!cur.empty() || pages.empty()) pages.push_back(cur);
    cur.clear();
    y = H - T;
  }
  void begin() { pages.clear(); cur.clear(); y = H - T; }
  void need(double h) { if (y - h < B) { pages.push_back(cur); cur.clear(); y = H - T; } }
  static std::string pesc(const std::string& s) {
    std::string o;
    for (char c : s) {
      if (c == '(' || c == ')' || c == '\\') o += '\\';
      o += c;
    }
    return o;
  }
  void text_at(double x, double yy, int font, double size, const std::string& t) {
    char b[96];
    snprintf(b, sizeof b, "BT /F%d %.1f Tf %.2f %.2f Td (", font, size, x, yy);
    cur += b + pesc(t) + ") Tj ET\n";
  }
  // runs wrapped into the width from x0; returns nothing (moves y)
  void flow(const std::vector<Run>& runs, double x0, double size, double lead, bool all_bold = false, double first_x = -1) {
    struct Word { std::string t; int font; bool space_after; };
    std::vector<Word> words;
    for (const Run& r : runs) {
      const int font = r.code ? 5 : ((r.b || all_bold) && r.i ? 4 : (r.b || all_bold) ? 2 : r.i ? 3 : 1);
      const std::string t = winansi(r.text);
      size_t a = 0;
      while (a < t.size()) {
        size_t e = t.find(' ', a);
        const bool sp = e != std::string::npos;
        if (!sp) e = t.size();
        if (e > a || sp) words.push_back({t.substr(a, e - a), font, sp});
        a = e + (sp ? 1 : 0);
      }
    }
    auto wof = [&](const std::string& t, int font) {
      double w = 0;
      for (unsigned char c : t) w += char_w(c, font == 2 || font == 4, font == 5);
      return w * size / 1000.0;
    };
    const double maxw = W - R - x0, spw = 278 * size / 1000.0;
    double x = first_x >= 0 ? first_x : x0;
    need(lead);
    bool line_has = false;
    for (const Word& w : words) {
      double ww = wof(w.t, w.font);
      if (line_has && x + ww > x0 + maxw) { y -= lead; need(lead); x = x0; line_has = false; }
      // (a single word wider than the line: cut by characters)
      std::string t = w.t;
      while (!line_has && ww > maxw && t.size() > 1) {
        size_t n = t.size();
        while (n > 1 && wof(t.substr(0, n), w.font) > maxw) --n;
        text_at(x, y, w.font, size, t.substr(0, n));
        t = t.substr(n);
        y -= lead;
        need(lead);
        ww = wof(t, w.font);
      }
      if (!t.empty()) text_at(x, y, w.font, size, t);
      x += ww + (w.space_after ? spw : 0);
      line_has = true;
    }
    y -= lead;
  }
  std::string finish(const std::string& title) {
    pages.push_back(cur);
    if (pages.size() > 1 && pages.front().empty()) pages.erase(pages.begin());
    // objects: 1 catalog, 2 pages, 3-7 fonts, 8 info, then per page: page, content
    std::vector<std::string> obj;
    std::string kids;
    const int n = (int)pages.size();
    for (int p = 0; p < n; ++p) kids += std::to_string(9 + 2 * p) + " 0 R ";
    obj.push_back("<< /Type /Catalog /Pages 2 0 R >>");
    obj.push_back("<< /Type /Pages /Kids [" + kids + "] /Count " + std::to_string(n) + " >>");
    for (const char* f : {"Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique", "Courier"})
      obj.push_back(std::string("<< /Type /Font /Subtype /Type1 /BaseFont /") + f + " /Encoding /WinAnsiEncoding >>");
    obj.push_back("<< /Title (" + pesc(winansi(title)) + ") /Producer (arcinferzero) >>");
    for (int p = 0; p < n; ++p) {
      std::string c = pages[p];
      char foot[128];
      snprintf(foot, sizeof foot, "BT /F1 8.5 Tf %.2f %.2f Td (%d / %d) Tj ET\n", W / 2 - 10, B / 2, p + 1, n);
      c += foot;
      char pg[256];
      snprintf(pg, sizeof pg, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] /Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R "
               "/F4 6 0 R /F5 7 0 R >> >> /Contents %d 0 R >>", W, H, 10 + 2 * p);
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
    for (size_t o : off) {
      char b[24];
      snprintf(b, sizeof b, "%010zu 00000 n \n", o);
      out += b;
    }
    out += "trailer\n<< /Size " + std::to_string(obj.size() + 1) + " /Root 1 0 R /Info 8 0 R >>\nstartxref\n" + std::to_string(xref) + "\n%%EOF\n";
    return out;
  }
};

inline std::string pdf(const std::string& title, const std::string& md) {
  Pdf p;
  p.begin();
  if (!trim(title).empty()) { p.flow({Run{title}}, p.L, 20, 26, true); p.y -= 6; }
  int last_num = 0;
  for (const Block& b : parse(md)) {
    switch (b.kind) {
      case Block::Heading: {
        const double sz[5] = {0, 16, 13.5, 12, 11};
        p.y -= b.level <= 2 ? 8 : 5;
        p.need(sz[b.level] * 2.6);  // (a heading not alone at the bottom of a page)
        p.flow(b.runs, p.L, sz[b.level], sz[b.level] * 1.35, true);
        p.y -= 2;
        break;
      }
      case Block::Para: p.flow(b.runs, p.L, 10.5, 14.5); p.y -= 5; break;
      case Block::Bullet:
      case Block::Number: {
        const double x = p.L + 14 + 16 * b.level;
        p.need(14.5);
        std::string mark;
        if (b.kind == Block::Bullet) mark = b.level % 2 ? "-" : "\x95";
        else { last_num = b.num ? b.num : last_num + 1; mark = std::to_string(last_num) + "."; }
        p.text_at(x - 12, p.y, 1, 10.5, mark);
        p.flow(b.runs, x + (b.kind == Block::Number ? 4 : 0), 10.5, 14.5);
        p.y -= 1.5;
        break;
      }
      case Block::Code: {
        p.y -= 2;
        for (const std::string& l : b.lines) {
          p.need(12.5);
          char rect[96];
          snprintf(rect, sizeof rect, "0.95 0.95 0.94 rg %.2f %.2f %.2f 12.5 re f 0 g\n", p.L - 4, p.y - 3.5, p.W - p.L - p.R + 8);
          p.cur += rect;
          std::string t = winansi(l);
          const size_t maxc = (size_t)((p.W - p.L - p.R) / (600 * 9.0 / 1000));
          if (t.size() > maxc) t = t.substr(0, maxc - 1) + ">";
          p.text_at(p.L, p.y, 5, 9, t);
          p.y -= 12.5;
        }
        p.y -= 6;
        break;
      }
      case Block::Rule: {
        p.need(10);
        char ln[96];
        snprintf(ln, sizeof ln, "0.6 G 0.5 w %.2f %.2f m %.2f %.2f l S 0 G\n", p.L, p.y + 4, p.W - p.R, p.y + 4);
        p.cur += ln;
        p.y -= 10;
        break;
      }
      case Block::Table: {
        size_t cols = 0;
        for (const auto& r : b.rows) cols = std::max(cols, r.size());
        if (!cols) break;
        const double tw = p.W - p.L - p.R, cw = tw / (double)cols, size = 9.5, lead = 12.5;
        for (size_t r = 0; r < b.rows.size(); ++r) {
          // each cell's text as wrapped lines, measured first (a row's height is its tallest cell)
          std::vector<std::vector<std::string>> cl(cols);
          size_t rows_h = 1;
          for (size_t c = 0; c < cols; ++c) {
            std::string t;
            if (c < b.rows[r].size()) for (const Run& x : b.rows[r][c]) t += x.text;
            t = winansi(t);
            std::string line;
            double lw = 0;
            const double maxw = cw - 8;
            size_t a = 0;
            while (a <= t.size()) {
              size_t e = t.find(' ', a);
              if (e == std::string::npos) e = t.size();
              const std::string w = t.substr(a, e - a);
              double ww = 0;
              for (unsigned char ch : w) ww += char_w(ch, r == 0, false) * size / 1000.0;
              if (!line.empty() && lw + ww > maxw) { cl[c].push_back(line); line.clear(); lw = 0; }
              line += (line.empty() ? "" : " ") + w;
              lw += ww + 278 * size / 1000.0;
              a = e + 1;
            }
            if (!line.empty()) cl[c].push_back(line);
            rows_h = std::max(rows_h, cl[c].size());
          }
          const double h = rows_h * lead + 6;
          p.need(h);
          char box[160];
          if (r == 0) {
            snprintf(box, sizeof box, "0.93 0.93 0.93 rg %.2f %.2f %.2f %.2f re f 0 g\n", p.L, p.y - h + lead, tw, h);
            p.cur += box;
          }
          for (size_t c = 0; c < cols; ++c) {
            snprintf(box, sizeof box, "0.6 G 0.5 w %.2f %.2f %.2f %.2f re S 0 G\n", p.L + c * cw, p.y - h + lead, cw, h);
            p.cur += box;
            for (size_t k = 0; k < cl[c].size(); ++k) p.text_at(p.L + c * cw + 4, p.y - 2 - k * lead, r == 0 ? 2 : 1, size, cl[c][k]);
          }
          p.y -= h;
        }
        p.y -= 8;
        break;
      }
    }
    if (b.kind != Block::Number) last_num = 0;
  }
  return p.finish(title);
}

}  // namespace doc
