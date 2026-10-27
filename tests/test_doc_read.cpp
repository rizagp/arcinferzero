// Checks of tools/doc_reader.h and tools/inflate.h: DOCX and PDF files written by Word (tests/data/doc_sample.docx,
// doc_sample.pdf: deflated parts and streams, ToUnicode fonts) and by tools/doc_writer.h (stored ZIP, standard PDF
// fonts) read back as paragraphs with headings and pages; the outline, chunks and BM25 search; damaged input.
//   test_doc_read.exe                 the checks
//   test_doc_read.exe FILE [QUERY]    a file's outline (and the chunks a query finds): to look at any document
#include "../tools/doc_reader.h"
#include "../tools/doc_writer.h"
#include <cstdio>
#include <fstream>
#include <sstream>

static int fails = 0;
static void check(const char* name, bool ok, const std::string& got = "") {
  printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name, ok || got.empty() ? "" : "  got: ", ok ? "" : got.c_str());
  fails += !ok;
}
static std::string slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
static bool has(const rdoc::Doc& d, const std::string& t, int level = -1) {
  for (const auto& p : d.paras)
    if (p.text.find(t) != std::string::npos && (level < 0 || p.level == level)) return true;
  return false;
}
static void show(const std::string& path, const std::string& query) {
  const rdoc::Index ix(rdoc::read(path, slurp(path)));
  printf("%zu paragraphs, %d pages, %zu chars, %zu chunks%s%s\n", ix.doc.paras.size(), ix.doc.pages, ix.doc.chars(),
         ix.chunks.size(), ix.doc.note.empty() ? "" : ", note: ", ix.doc.note.c_str());
  for (const auto& s : ix.outline)
    printf("%*s%s  (p. %d, %zu chars)\n", (s.level - 1) * 2, "", s.title.c_str(), s.page, s.chars);
  if (query.empty()) {
    for (size_t k = 0; k < ix.doc.paras.size() && k < 25; ++k)
      printf("[%d|%d] %s\n", ix.doc.paras[k].page, ix.doc.paras[k].level, rdoc::Index::cut_words(ix.doc.paras[k].text, 150).c_str());
    return;
  }
  for (const auto& h : ix.search(query, 5))
    printf("--- %.2f p.%d-%d %s\n%s\n", h.score, ix.chunks[h.chunk].page_from, ix.chunks[h.chunk].page_to,
           ix.outline[ix.chunks[h.chunk].section].title.c_str(), rdoc::Index::cut_words(ix.chunks[h.chunk].text, 400).c_str());
}

// a long made-up document (chapters of varied sentences, one fact to find): for timing and search on many pages
static std::string long_markdown(int chapters) {
  static const char* w[] = {"laporan", "kegiatan", "anggaran", "pelayanan", "peserta", "program", "evaluasi", "target",
                            "realisasi", "wilayah", "kantor", "kinerja", "pengawasan", "data", "sistem", "rencana",
                            "kebijakan", "pelaksanaan", "monitoring", "risiko", "mitigasi", "pengendalian", "dokumen"};
  uint32_t r = 12345;
  auto next = [&] { r = r * 1103515245u + 12345u; return (r >> 16) % 23; };
  std::string md;
  for (int c = 1; c <= chapters; ++c) {
    md += "# Bab " + std::to_string(c) + " " + w[next()] + " " + w[next()] + "\n\n";
    for (int s = 1; s <= 3; ++s) {
      md += "## " + std::to_string(c) + "." + std::to_string(s) + " " + w[next()] + "\n\n";
      for (int p = 0; p < 3; ++p) {
        for (int k = 0; k < 9; ++k) {
          std::string sent = w[next()];
          sent[0] = (char)toupper((unsigned char)sent[0]);
          for (int j = 0; j < 9; ++j) sent += std::string(" ") + w[next()];
          md += sent + ". ";
        }
        if (c == chapters * 2 / 3 && s == 2 && p == 1) md += "Server cadangan di gudang Cikarang dipindahkan ke Batam pada 14 Maret. ";
        md += "\n\n";
      }
    }
  }
  return md;
}

// counts only (no text): words, words run together (> 25 letters, no URL or path), one-letter words (spaces inside
// words), characters that did not decode (U+FFFD, '?' between letters), headings by level
static void quality(const std::string& path) {
  const rdoc::Doc d = rdoc::read(path, slurp(path));
  size_t words = 0, glued = 0, single = 0, bad = 0, levels[7] = {};
  for (const auto& p : d.paras) {
    levels[std::min(p.level, 6)]++;
    std::string w;
    auto end = [&] {
      if (!w.empty()) {
        ++words;
        const bool link = w.find("://") != std::string::npos || w.find('\\') != std::string::npos || w.find('/') != std::string::npos ||
                          w.find('_') != std::string::npos || w.find('@') != std::string::npos;
        size_t letters = 0;
        for (unsigned char c : w) letters += isalpha(c) != 0;
        if (letters > 25 && !link) ++glued;
        if (w.size() == 1 && isalpha((unsigned char)w[0]) && w != "a" && w != "A" && w != "I") ++single;
      }
      w.clear();
    };
    for (size_t i = 0; i < p.text.size(); ++i) {
      const char c = p.text[i];
      if (p.text.compare(i, 3, "\xef\xbf\xbd") == 0) ++bad;
      if (c == '?' && i > 0 && i + 1 < p.text.size() && isalpha((unsigned char)p.text[i - 1]) && isalpha((unsigned char)p.text[i + 1])) ++bad;
      if (c == ' ') end();
      else w += c;
    }
    end();
  }
  printf("%zu words | run together %zu (%.2f%%) | one-letter %zu (%.2f%%) | undecoded %zu | headings h1 %zu h2 %zu h3 %zu h4+ %zu%s%s\n",
         words, glued, words ? 100.0 * glued / words : 0, single, words ? 100.0 * single / words : 0, bad, levels[1], levels[2],
         levels[3], levels[4] + levels[5] + levels[6], d.note.empty() ? "" : " | note: ", d.note.c_str());
}

int main(int argc, char** argv) {
  if (argc > 2 && std::string(argv[1]) == "--quality") {
    quality(argv[2]);
    return 0;
  }
  if (argc > 2 && std::string(argv[1]) == "--make-long") {
    doc::Options o;
    o.title = "Dokumen uji panjang";
    const std::string md = long_markdown(120);
    std::ofstream(std::string(argv[2]) + ".docx", std::ios::binary) << doc::docx(md, o);
    std::ofstream(std::string(argv[2]) + ".pdf", std::ios::binary) << doc::pdf(md, o);
    std::ofstream(std::string(argv[2]) + ".md", std::ios::binary) << md;
    return 0;
  }
  if (argc > 1) {
    show(argv[1], argc > 2 ? argv[2] : "");
    return 0;
  }
  const std::string dir = std::string(argv[0]).substr(0, std::string(argv[0]).find_last_of("\\/") + 1) + "..\\tests\\data\\";
  // inflate: a stored block and a fixed-code block (Python's zlib, levels 0 and 1); dynamic blocks: the Word files
  {
    std::string out;
    const unsigned char stored[] = {0x01, 0x05, 0x00, 0xfa, 0xff, 'h', 'e', 'l', 'l', 'o'};
    check("inflate: stored block", infl::inflate(stored, sizeof stored, out) && out == "hello", out);
    out.clear();
    const unsigned char fixed[] = {0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x40, 0x22, 0x01};  // ("hello hello hello")
    check("inflate: fixed codes with a back reference", infl::inflate(fixed, sizeof fixed, out) && out == "hello hello hello", out);
    out.clear();
    const unsigned char cut[] = {0xcb, 0x48, 0xcd};
    check("inflate: a cut stream fails", !infl::inflate(cut, sizeof cut, out));
    std::string bomb;
    check("inflate: max_out stops it", !infl::inflate(fixed, sizeof fixed, bomb, 8));
  }
  // documents written by doc_writer.h (no compression): every format back as text
  {
    const std::string md = "# Laporan Bulanan\n\nPendahuluan singkat tentang migrasi server.\n\n## Anggaran\n\n| Pos | Nilai |\n|---|---:|\n| Server | 12 juta |\n\n## Jadwal\n\nRapat berikut Senin 09:00.\n";
    doc::Options o;
    o.title = "Uji";
    const rdoc::Doc dx = rdoc::read("a.docx", doc::docx(md, o));
    check("doc_writer DOCX: headings", has(dx, "Anggaran", 2) && has(dx, "Laporan Bulanan", 1));
    check("doc_writer DOCX: table row", has(dx, "Server | 12 juta"));
    const rdoc::Doc dp = rdoc::read("a.pdf", doc::pdf(md, o));
    check("doc_writer PDF: text", has(dp, "Pendahuluan singkat tentang migrasi server."));
    check("doc_writer PDF: a heading by its size", has(dp, "Anggaran") && [&] {
      for (const auto& p : dp.paras) if (p.text == "Anggaran") return p.level > 0;
      return false; }());
  }
  // files written by Word
  {
    const std::string docx = slurp(dir + "doc_sample.docx");
    check("Word DOCX fixture present", !docx.empty());
    const rdoc::Doc d = rdoc::read("doc_sample.docx", docx);
    check("Word DOCX: paragraphs", d.paras.size() > 10, std::to_string(d.paras.size()));
    check("Word DOCX: headings found", std::count_if(d.paras.begin(), d.paras.end(), [](const rdoc::Para& p) { return p.level > 0; }) >= 3);
    const std::string pdfs = slurp(dir + "doc_sample.pdf");
    check("Word PDF fixture present", !pdfs.empty());
    const rdoc::Doc p = rdoc::read("doc_sample.pdf", pdfs);
    check("Word PDF: pages", p.pages >= 2, std::to_string(p.pages));
    check("Word PDF: paragraphs", p.paras.size() > 10, std::to_string(p.paras.size()));
    check("Word PDF: headings found", std::count_if(p.paras.begin(), p.paras.end(), [](const rdoc::Para& x) { return x.level > 0; }) >= 3);
    // the same words in both: the PDF's text is the document's text (spaces where Word put them)
    size_t same = 0, total = 0;
    std::set<std::string> pw;
    for (const auto& x : p.paras) for (const std::string& w : rdoc::words(x.text)) pw.insert(w);
    for (const auto& x : d.paras) for (const std::string& w : rdoc::words(x.text)) { ++total; same += pw.count(w); }
    check("Word PDF: 95% of the DOCX's words", total && same * 100 >= total * 95, std::to_string(same) + " / " + std::to_string(total));
    const rdoc::Index ix(p);
    check("outline from the headings", ix.by_headings && ix.outline.size() >= 3, std::to_string(ix.outline.size()));
    check("chunks within their size", std::all_of(ix.chunks.begin(), ix.chunks.end(), [](const rdoc::Chunk& c) { return c.text.size() <= 3700; }));
  }
  // search: the chunk with the words first; Indonesian particles
  {
    std::string md;
    for (int k = 1; k <= 30; ++k) md += "## Bagian " + std::to_string(k) + "\n\nIsi umum bagian ini tentang kegiatan rutin kantor dan administrasi.\n\n";
    md += "## Keamanan\n\nPemeriksaan kata sandi dan enkripsi cadangannya dilakukan setiap Jumat.\n\n";
    const rdoc::Index ix(rdoc::read("x.md", md));
    const auto h = ix.search("kapan enkripsi cadangan dilakukan?", 3);
    check("search: the right chunk first", !h.empty() && ix.chunks[h[0].chunk].text.find("enkripsi") != std::string::npos);
    check("search: a word with -nya matches", !ix.search("cadangan", 1).empty());
    check("search: nothing for unknown words", ix.search("zebra", 3).empty());
  }
  // damaged and odd input: no crash, a note
  {
    check("damaged DOCX: a note", !rdoc::read("x.docx", "PK\x03\x04garbage").note.empty());
    check("damaged PDF: no crash", rdoc::read("x.pdf", "%PDF-1.4\n1 0 obj << /Type /Page /Contents 2 0 R >> endobj\n2 0 obj << /Length 999999 >> stream\nBT (x) Tj").paras.size() <= 1);
    // hostile shapes: long runs of delimiters, deep nesting, a stream that inflates far, objects pointing at themselves
    std::string evil = "%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [2 0 R 3 0 R] >> endobj\n"
                       "3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R /Resources << /XObject << /X 5 0 R >> >> >> endobj\n"
                       "4 0 obj << /Length 20 >> stream\n/X Do /X Do /X Do\nendstream endobj\n"
                       "5 0 obj << /Subtype /Form /Resources << /XObject << /X 5 0 R >> >> /Length 6 >> stream\n/X Do\nendstream endobj\n";
    evil += std::string(200000, ']') + std::string(100000, '>') + std::string(5000, '[') + std::string(5000, '<') + "<<";
    check("hostile PDF: no crash", rdoc::read("x.pdf", evil).paras.empty());
    check("binary .doc: a note", rdoc::read("x.doc", std::string("\xd0\xcf\x11\xe0\0\0", 6)).note.find("DOCX") != std::string::npos);
    check("ANSI text read as Windows-1252", rdoc::read("x.txt", "caf\xe9").paras[0].text == "caf\xc3\xa9");
  }
  printf(fails ? "test_doc_read: %d FAILED\n" : "test_doc_read: all passed\n", fails);
  return fails ? 1 : 0;
}
