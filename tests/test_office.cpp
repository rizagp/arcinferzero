// Checks of tools/xlsx.h and tools/pptx.h. Excel: a workbook saved by Excel (tests/data/xlsx_sample.xlsx: dates,
// percentages, a currency format, formulas with cached values, rich text, merged cells, an empty sheet) reads as headed
// rows; workbooks written from Markdown read back (numbers, Rp amounts, percentages, dates as real values, Indonesian
// or US number style, sheet names); the XML element order Excel needs; CSV; hostile input (huge references, bad
// shared string indexes, deep nesting, too many cells). PowerPoint: a deck saved by PowerPoint
// (tests/data/pptx_sample.pptx: titles, bullet levels, a table, speaker notes, alt text, a hidden slide) reads as one
// heading per slide; SmartArt, merged table cells and odd or hostile parts.
//   test_office.exe                 the checks
//   test_office.exe --write FILE    the sample workbook from Markdown (to open in Excel: tests/run_tests.ps1 does not)
#include "../tools/doc_reader.h"
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
    if (p.text == t && (level < 0 || p.level == level)) return true;
  return false;
}
static std::string all(const rdoc::Doc& d) {
  std::string o;
  for (const auto& p : d.paras) o += p.text + "\n";
  return o;
}
// a workbook from hand-written parts (sheet XML, shared strings, styles): for odd and hostile input
static std::string book(const std::vector<std::string>& sheets, const std::string& sst = "", const std::string& styles = "") {
  std::vector<std::pair<std::string, std::string>> p;
  std::string wb = "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
                   "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets>",
              rels = "<Relationships>";
  for (size_t k = 0; k < sheets.size(); ++k) {
    const std::string n = std::to_string(k + 1);
    wb += "<sheet name=\"S" + n + "\" sheetId=\"" + n + "\" r:id=\"rId" + n + "\"/>";
    rels += "<Relationship Id=\"rId" + n + "\" Target=\"worksheets/sheet" + n + ".xml\"/>";
    p.push_back({"xl/worksheets/sheet" + n + ".xml", sheets[k]});
  }
  p.push_back({"xl/workbook.xml", wb + "</sheets></workbook>"});
  p.push_back({"xl/_rels/workbook.xml.rels", rels + "</Relationships>"});
  if (!sst.empty()) p.push_back({"xl/sharedStrings.xml", sst});
  if (!styles.empty()) p.push_back({"xl/styles.xml", styles});
  return doc::zip(p);
}

static const char* kSample =
    "# Laporan Nilai\n\nRingkasan nilai ujian semester ganjil.\n\n"
    "## Nilai Siswa\n\n| Nama | Tanggal | Nilai | Harga | Diskon | Telepon | Rumus |\n|---|---|---:|---:|---:|---|---|\n"
    "| Budi | 2026-10-27 | 1.234,5 | Rp 1.234.000 | 15 % | 0812345678 | =1+1 |\n"
    "| **Total** | 27/10/2026 14:30 | 12.500 | $12.50 | 12,5% | 12 | -3,25 |\n\n"
    "- catatan satu\n- catatan dua\n\n"
    "## Data [Q3]: a/b*c? dengan nama yang sangat panjang sekali\n\n| A | B |\n|---|---|\n| x | 1 |\n\n"
    "## Nilai Siswa\n\n| A | B |\n|---|---|\n| y | 2 |\n";

int main(int argc, char** argv) {
  if (argc > 2 && std::string(argv[1]) == "--write") {
    doc::Options o;
    o.title = "Laporan Nilai";
    std::ofstream(argv[2], std::ios::binary) << xlsx::write(kSample, o);
    return 0;
  }
  const std::string dir = std::string(argv[0]).substr(0, std::string(argv[0]).find_last_of("\\/") + 1) + "..\\tests\\data\\";
  // a workbook saved by Excel
  {
    const std::string x = slurp(dir + "xlsx_sample.xlsx");
    check("Excel fixture present", !x.empty());
    const rdoc::Doc d = rdoc::read("xlsx_sample.xlsx", x);
    check("Excel: no note", d.note.empty(), d.note);
    check("Excel: sheet headings in order", d.paras.size() > 3 && d.paras[0].text == "Sheet: Data" && d.paras[0].level == 1 && has(d, "Sheet: Rekap Q3", 1) && has(d, "Sheet: Kosong", 1));
    check("Excel: header row", has(d, "Nama | Tanggal | Nilai | Persen | Harga | Lulus | Catatan | Waktu"), all(d));
    check("Excel: dates, percent, currency, boolean, date and time",
          has(d, "Nama: Budi | Tanggal: 2026-10-27 | Nilai: 80 | Persen: 15.5% | Harga: Rp 1234000 | Lulus: TRUE | Catatan: lulus dengan pujian | Waktu: 2026-10-27 14:30"), all(d));
    check("Excel: rich text runs joined, an empty cell left out",
          has(d, "Nama: Siti | Tanggal: 2025-01-05 | Nilai: 92.5 | Persen: 20.0% | Harga: Rp 750000 | Lulus: TRUE | Catatan: nilai tertinggi | Waktu: 2025-01-05 08:05") ||
          has(d, "Nama: Siti | Tanggal: 2025-01-05 | Nilai: 92.5 | Persen: 20% | Harga: Rp 750000 | Lulus: TRUE | Catatan: nilai tertinggi | Waktu: 2025-01-05 08:05"), all(d));
    check("Excel: a leap day, midnight without a time", has(d, "Nama: Andi | Tanggal: 2024-02-29 | Nilai: 55 | Persen: 5.0% | Harga: Rp 99 | Lulus: FALSE | Catatan: remedial | Waktu: 2024-02-29") ||
          has(d, "Nama: Andi | Tanggal: 2024-02-29 | Nilai: 55 | Persen: 5% | Harga: Rp 99 | Lulus: FALSE | Catatan: remedial | Waktu: 2024-02-29"), all(d));
    // (the TEXT() result is what Excel computed and stored, in an Indonesian locale: "0.0" read as a thousands format)
    check("Excel: formulas as their cached values, an error", has(d, "Nama: Total | Nilai: 227.5 | Persen: #DIV/0! | Catatan: rata-rata 76"), all(d));
    check("Excel: a merged title above the header", has(d, "Laporan Kuartal 3") && has(d, "Wilayah | Juli | Agustus | September"), all(d));
    check("Excel: merged cells, a column without a header", has(d, "Wilayah: Jawa Timur | Juli: 7 | Agustus: 9") && has(d, "catatan jauh"), all(d));
    check("Excel: an empty sheet", has(d, "(empty)"));
    const rdoc::Index ix(d);
    const auto h = ix.search("nilai Andi remedial", 3);
    check("Excel: search finds the row with its column names", !h.empty() && ix.chunks[h[0].chunk].text.find("Catatan: remedial") != std::string::npos);
  }
  // written from Markdown, read back
  {
    doc::Options o;
    o.title = "Laporan Nilai";
    const std::string z = xlsx::write(kSample, o);
    const rdoc::Doc d = rdoc::read("a.xlsx", z);
    check("write: Notes sheet first", d.paras.size() > 2 && d.paras[0].text == "Sheet: Notes" && has(d, "Ringkasan nilai ujian semester ganjil."), all(d));
    check("write: a table's sheet named by its heading", has(d, "Sheet: Nilai Siswa", 1), all(d));
    check("write: numbers, Rp, percent, date; a phone number and a formula stay text",
          has(d, "Nama: Budi | Tanggal: 2026-10-27 | Nilai: 1234.5 | Harga: Rp 1234000 | Diskon: 15% | Telepon: 0812345678 | Rumus: =1+1"), all(d));
    check("write: date with time, thousands, dollars, decimal comma, negative",
          has(d, "Nama: Total | Tanggal: 2026-10-27 14:30 | Nilai: 12500 | Harga: $12.5 | Diskon: 12.5% | Telepon: 12 | Rumus: -3.25"), all(d));
    check("write: list items in Notes", has(d, "\xE2\x80\xA2 catatan satu"));
    bool clean = false, dup = false;
    for (const auto& p : d.paras) {
      if (p.level != 1 || p.text.rfind("Sheet: Data", 0) != 0) { dup = dup || p.text == "Sheet: Nilai Siswa (2)"; continue; }
      const std::string n = p.text.substr(7);
      clean = n.find_first_of("[]:*?/\\") == std::string::npos && n.size() <= 31;
    }
    check("write: sheet name without [ ] : * ? / \\, 31 characters", clean, all(d));
    check("write: a repeated name gets (2)", dup, all(d));
    std::string sheet, wb, st;
    infl::zip_entry(z, "xl/worksheets/sheet2.xml", sheet);
    infl::zip_entry(z, "xl/workbook.xml", wb);
    infl::zip_entry(z, "xl/styles.xml", st);
    auto before = [&](const std::string& s, std::vector<const char*> order) {
      size_t at = 0;
      for (const char* t : order) {
        const size_t p = s.find(t);
        if (p == std::string::npos || p < at) return false;
        at = p;
      }
      return true;
    };
    check("write: worksheet element order", before(sheet, {"<dimension", "<sheetViews", "<pane", "<selection", "<sheetFormatPr", "<cols", "<sheetData", "<autoFilter", "<pageMargins"}));
    check("write: frozen header, the data sheet selected", sheet.find("state=\"frozen\"") != std::string::npos && sheet.find("tabSelected=\"1\"") != std::string::npos &&
          wb.find("activeTab=\"1\"") != std::string::npos);
    check("write: styles element order", before(st, {"<numFmts", "<fonts", "<fills", "<borders", "<cellStyleXfs", "<cellXfs", "<cellStyles"}));
    check("write: workbook element order and the filter's name", before(wb, {"<workbookPr", "<bookViews", "<sheets", "<definedNames"}) &&
          wb.find("_xlnm._FilterDatabase") != std::string::npos);
    // Indonesian or US numbers: each table by its numbers that show the style, else the document's, else Indonesian
    {
      const std::string mixed = "## US\n\n| Item | Qty | Price |\n|---|---|---|\n| A | 1,500 | 3.75 |\n| B | 2,000 | 1,234.50 |\n\n"
                                "## ID\n\n| Item | Qty | Harga |\n|---|---|---|\n| A | 1.500 | 2,5 |\n\n"
                                "## Ambigu\n\n| Item | Qty |\n|---|---|\n| A | 1,500 |\n| B | 1.500 |\n";
      const rdoc::Doc a = rdoc::read("m.xlsx", xlsx::write(mixed, doc::Options{}));
      check("numbers auto: a US table", has(a, "Item: A | Qty: 1500 | Price: 3.75") && has(a, "Item: B | Qty: 2000 | Price: 1234.5"), all(a));
      check("numbers auto: an Indonesian table in the same file", has(a, "Item: A | Qty: 1500 | Harga: 2.5"), all(a));
      // (the document: 3.75 and 1,234.50 US against 2,5 Indonesian)
      check("numbers auto: a table that cannot tell goes by the document", has(a, "Item: A | Qty: 1500") && has(a, "Item: B | Qty: 1.5"), all(a));
      const rdoc::Doc n = rdoc::read("n.xlsx", xlsx::write("| Item | Qty |\n|---|---|\n| A | 1,500 |\n| B | 1.500 |\n", doc::Options{}));
      check("numbers auto: nothing tells, Indonesian", has(n, "Item: A | Qty: 1.5") && has(n, "Item: B | Qty: 1500"), all(n));
      const std::string amb = "| Item | Qty |\n|---|---|\n| A | 1,500 |\n| B | 1.500 |\n";
      const rdoc::Doc us = rdoc::read("u.xlsx", xlsx::write(amb, doc::Options{}, "us")), id = rdoc::read("i.xlsx", xlsx::write(amb, doc::Options{}, "id"));
      check("numbers forced us", has(us, "Item: A | Qty: 1500") && has(us, "Item: B | Qty: 1.5"), all(us));
      check("numbers forced id", has(id, "Item: A | Qty: 1.5") && has(id, "Item: B | Qty: 1500"), all(id));
      check("group_of", xlsx::group_of("1.234,5") == '.' && xlsx::group_of("1,234.5") == ',' && xlsx::group_of("1.234.567") == '.' &&
            xlsx::group_of("2,5") == '.' && xlsx::group_of("3.14") == ',' && xlsx::group_of("1.500") == 0 && xlsx::group_of("12") == 0);
    }
    // only tables: no Notes sheet; only text: the Notes sheet alone
    const rdoc::Doc t = rdoc::read("b.xlsx", xlsx::write("## Stok\n\n| Barang | Jumlah |\n|---|---|\n| Pena | 1.500 |\n", doc::Options{}));
    check("write: only a table, no Notes", t.paras.size() == 3 && t.paras[0].text == "Sheet: Stok" && t.paras[2].text == "Barang: Pena | Jumlah: 1500", all(t));
    const rdoc::Doc n = rdoc::read("c.xlsx", xlsx::write("Hanya teks.\n", doc::Options{}));
    check("write: only text, a Notes sheet", n.paras.size() == 2 && n.paras[1].text == "Hanya teks.", all(n));
    check("write: empty content, still a workbook", rdoc::read("d.xlsx", xlsx::write("", doc::Options{})).note.empty());
  }
  // cell values the writer recognizes
  {
    double v = 0;
    std::string f;
    check("number: 1.234,5", xlsx::number("1.234,5", v, f) && v == 1234.5 && f == "#,##0.0");
    check("number: 1,234.5", xlsx::number("1,234.5", v, f) && v == 1234.5);
    check("number: 1.500 (thousands)", xlsx::number("1.500", v, f) && v == 1500);
    check("number: 3.14, 0.125", xlsx::number("3.14", v, f) && v == 3.14 && xlsx::number("0.125", v, f) && v == 0.125 && f.empty());
    check("number: 2,5 (decimal comma)", xlsx::number("2,5", v, f) && v == 2.5);
    check("number: Rp 1.234.000", xlsx::number("Rp 1.234.000", v, f) && v == 1234000 && f == "\"Rp \"#,##0");
    check("number: Rp1,500 (currency: thousands)", xlsx::number("Rp1,500", v, f) && v == 1500);
    check("number: -Rp 5.000 and (U+2212)7", xlsx::number("-Rp 5.000", v, f) && v == -5000 && xlsx::number("\xE2\x88\x92" "7", v, f) && v == -7);
    check("number: 15 %", xlsx::number("15 %", v, f) && std::fabs(v - 0.15) < 1e-12 && f == "0%");
    check("number: text stays text", !xlsx::number("12 juta", v, f) && !xlsx::number("1.2.3,4,5", v, f) && !xlsx::number("1,23,456", v, f) &&
          !xlsx::number("007", v, f) && !xlsx::number("1234567890123456", v, f) && !xlsx::number("Rp", v, f) && !xlsx::number("1e5", v, f));
    check("date: ISO and day first", xlsx::date("2026-10-27", v, f) && v == 46322 && f == "yyyy-mm-dd" && xlsx::date("27/10/2026", v, f) && v == 46322 &&
          f == "dd/mm/yyyy" && xlsx::date("27-10-2026", v, f) && v == 46322);
    check("number: phone and account numbers stay text", !xlsx::number("+6281234567890", v, f) && !xlsx::number("123401000123456", v, f) &&
          !xlsx::number("628123456789", v, f) && xlsx::number("12345678901", v, f) && xlsx::number("+5", v, f) && v == 5);
    check("date: a US table reads 03/04/2026 as March 4", xlsx::date("03/04/2026", v, f, true) && v == 46085 &&
          xlsx::date("03/04/2026", v, f) && v == 46115 && xlsx::date("2026-03-04", v, f, true) && v == 46085);
    check("elapsed time: 1.25 days as 30:00", xlsx::value_text(1.25, xlsx::fmt_of(164, "[h]:mm"), false) == "30:00" &&
          xlsx::value_text(1.5 / 24 + 2, xlsx::fmt_of(46, ""), false) == "49:30" && xlsx::fmt_of(46, "").kind == xlsx::Fmt::Elapsed);
    check("date: with a time", xlsx::date("2026-10-27T12:00", v, f) && v == 46322.5 && f == "yyyy-mm-dd hh:mm");
    check("date: not a date", !xlsx::date("2026-02-30", v, f) && !xlsx::date("31/04/2026", v, f) && !xlsx::date("2025-02-29", v, f) &&
          !xlsx::date("1899-12-31", v, f) && !xlsx::date("10:30", v, f) && !xlsx::date("2026-10-27 25:00", v, f));
    check("serial dates", xlsx::serial_text(46322, false, false) == "2026-10-27" && xlsx::serial_text(60, false, false) == "1900-02-29" &&
          xlsx::serial_text(61, false, false) == "1900-03-01" && xlsx::serial_text(1, false, false) == "1900-01-01" &&
          xlsx::serial_text(0.75, true, false) == "18:00" && xlsx::serial_text(44860, false, true) == "2026-10-27");
    check("formats", xlsx::fmt_of(14, "").kind == xlsx::Fmt::Date && xlsx::fmt_of(164, "\"Total\" 0").kind == xlsx::Fmt::Number &&
          xlsx::fmt_of(164, "[$-421]dd mmmm yyyy").kind == xlsx::Fmt::Date && xlsx::fmt_of(164, "[h]:mm").kind == xlsx::Fmt::Elapsed &&
          xlsx::fmt_of(164, "0.0%").kind == xlsx::Fmt::Percent && xlsx::fmt_of(164, "[$Rp-421] #,##0").prefix == "Rp " &&
          xlsx::fmt_of(164, "General").kind == xlsx::Fmt::Number && xlsx::fmt_of(164, "#,##0;[Red]-#,##0").kind == xlsx::Fmt::Number);
  }
  // odd but valid input: inline strings, prefixed tags, cells without references, values copied into merged cells, a
  // style index past the styles, single-quoted attributes
  {
    const std::string sh = "<x:worksheet xmlns:x=\"main\"><x:sheetData><x:row r=\"1\"><x:c t=\"inlineStr\"><x:is><x:t>Kota</x:t></x:is></x:c>"
                           "<x:c t='inlineStr'><x:is><x:r><x:t>Jum</x:t></x:r><x:r><x:t>lah</x:t></x:r></x:is></x:c></x:row>"
                           "<x:row><x:c><x:v>1</x:v></x:c><x:c s=\"99\"><x:v>2</x:v></x:c></x:row>"
                           "<x:row r=\"5\"><x:c r=\"A5\" t=\"s\"><x:v>0</x:v></x:c><x:c r=\"B5\" t=\"s\"><x:v>0</x:v></x:c></x:row></x:sheetData>"
                           "<x:mergeCells><x:mergeCell ref=\"A5:B5\"/></x:mergeCells></x:worksheet>";
    const rdoc::Doc d = rdoc::read("o.xlsx", book({sh}, "<sst><si><t>Bandung</t></si></sst>"));
    check("odd: inline strings, prefixes, rows without r", has(d, "Kota | Jumlah") && has(d, "Kota: 1 | Jumlah: 2"), all(d));
    check("odd: a merged range keeps its first cell", has(d, "Kota: Bandung"), all(d));
  }
  // hostile input: no crash, bounded time and memory
  {
    check("hostile: not a ZIP", !rdoc::read("x.xlsx", "PK\x03\x04garbage").note.empty());
    check("hostile: binary .xls", rdoc::read("x.xls", std::string("\xd0\xcf\x11\xe0\0\0\0\0", 8)).note.find("XLSX") != std::string::npos);
    const std::string z = book({"<worksheet/>"});
    check("hostile: a cut ZIP", rdoc::read("x.xlsx", z.substr(0, z.size() / 2)).paras.size() <= 2);
    std::string sh = "<worksheet><dimension ref=\"A1:XFD1048576\"/><sheetData><row r=\"1\"><c r=\"A1\" t=\"s\"><v>5</v></c>"
                     "<c r=\"B1\" t=\"s\"><v>-1</v></c><c r=\"C1\" t=\"s\"><v>x</v></c><c r=\"D1\" t=\"s\"><v>99999999999999999999</v></c>"
                     "<c r=\"E1\" s=\"-5\"><v>3</v></c><c r=\"XFD1048576\"><v>7</v></c><c r=\"XFE1\"><v>8</v></c><c r=\"A99999999\"><v>9</v></c>"
                     "<c r=\"F1\" t=\"n\"><v>1e308</v></c><c r=\"G1\" s=\"1\"><v>-1e308</v></c></row></sheetData>"
                     "<mergeCells>";
    for (int k = 0; k < 20000; ++k) sh += "<mergeCell ref=\"A1:XFD1048576\"/>";
    sh += "</mergeCells></worksheet>";
    const std::string styles = "<styleSheet><cellXfs><xf numFmtId=\"0\"/><xf numFmtId=\"14\"/></cellXfs></styleSheet>";
    const rdoc::Doc d = rdoc::read("x.xlsx", book({sh}, "<sst><si><t>a</t></si></sst>", styles));
    check("hostile: bad indexes and huge references", d.paras.size() >= 2, all(d));
    std::string deep = "<worksheet>";
    for (int k = 0; k < 200000; ++k) deep += "<a>";
    deep += "<sheetData><row><c><v>42</v></c></row></sheetData>";
    for (int k = 0; k < 200000; ++k) deep += "</a>";
    check("hostile: deep nesting", has(rdoc::read("x.xlsx", book({deep + "</worksheet>"})), "42"));
    check("hostile: an unclosed tag, a quote never closed", rdoc::read("x.xlsx", book({"<worksheet><sheetData><row><c r=\"A1><v>1"})).paras.size() <= 2);
    std::string big = "<worksheet><sheetData>";
    for (int r = 1; r <= 26000; ++r) {
      big += "<row r=\"" + std::to_string(r) + "\">";
      for (int c = 0; c < 10; ++c) big += "<c><v>" + std::to_string(c) + "</v></c>";
      big += "</row>";
    }
    big += "</sheetData></worksheet>";
    const rdoc::Doc b = rdoc::read("x.xlsx", book({big, "<worksheet><sheetData><row><c><v>1</v></c></row></sheetData></worksheet>"}));
    check("hostile: 260,000 cells cut at the limit with a note", b.note.find("200000") != std::string::npos && b.paras.size() < 20100 && has(b, "(not read: the cell limit was reached)"), b.note);
  }
  // a presentation saved by PowerPoint
  {
    const std::string x = slurp(dir + "pptx_sample.pptx");
    check("PowerPoint fixture present", !x.empty());
    const rdoc::Doc d = rdoc::read("pptx_sample.pptx", x);
    check("PowerPoint: five slides as pages, no note", d.pages == 5 && d.note.empty(), d.note);
    check("PowerPoint: slide titles as headings", has(d, "Slide 1: Laporan Kuartal 3", 1) && has(d, "Slide 2: Ringkasan", 1) && has(d, "Slide 5: Proses", 1), all(d));
    check("PowerPoint: subtitle, bullets with their level", has(d, "Tim Data, Oktober 2026") && has(d, "- Penjualan naik 12 %") && has(d, "  - Jawa Barat paling tinggi"), all(d));
    check("PowerPoint: speaker notes", has(d, "Notes: Sebutkan target Q4 sebesar 15 %"), all(d));
    check("PowerPoint: a table with its header", has(d, "Wilayah: Jawa Timur | Juli: 7 | Agustus: 9"), all(d));
    check("PowerPoint: a picture's alt text", has(d, "[image: grafik penjualan per wilayah]"), all(d));
    check("PowerPoint: a slide without a title, hidden", has(d, "Slide 4: Catatan penutup tanpa judul (hidden)", 1), all(d));
    bool pages = true;
    for (const auto& p : d.paras) pages = pages && p.page >= 1 && p.page <= 5;
    check("PowerPoint: each line on its slide's page", pages && d.paras.back().page == 5);
    const rdoc::Index ix(d);
    check("PowerPoint: the outline is the slides", ix.by_headings && ix.outline.size() == 5, std::to_string(ix.outline.size()));
  }
  // presentations from hand-written parts: SmartArt, merged table cells, placeholders left out, odd and hostile input
  {
    auto deck = [](const std::vector<std::string>& slides, const std::vector<std::pair<std::string, std::string>>& extra = {}) {
      std::vector<std::pair<std::string, std::string>> p;
      std::string pres = "<p:presentation><p:sldIdLst>", rels = "<Relationships>";
      for (size_t k = 0; k < slides.size(); ++k) {
        const std::string n = std::to_string(k + 1);
        pres += "<p:sldId id=\"" + std::to_string(256 + k) + "\" r:id=\"rId" + n + "\"/>";
        rels += "<Relationship Id=\"rId" + n + "\" Type=\"x/slide\" Target=\"slides/slide" + n + ".xml\"/>";
        p.push_back({"ppt/slides/slide" + n + ".xml", slides[k]});
      }
      p.push_back({"ppt/presentation.xml", pres + "</p:sldIdLst></p:presentation>"});
      p.push_back({"ppt/_rels/presentation.xml.rels", rels + "</Relationships>"});
      for (const auto& e : extra) p.push_back(e);
      return doc::zip(p);
    };
    const std::string s1 = "<p:sld><p:cSld><p:spTree>"
                           "<p:sp><p:nvSpPr><p:nvPr><p:ph type=\"title\"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Alur &amp; Tahap</a:t></a:r></a:p></p:txBody></p:sp>"
                           "<p:sp><p:nvSpPr><p:nvPr><p:ph idx=\"1\"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:pPr><a:buNone/></a:pPr><a:r><a:t>tanpa</a:t></a:r>"
                           "<a:br/><a:r><a:t>bullet</a:t></a:r></a:p></p:txBody></p:sp>"
                           "<p:sp><p:nvSpPr><p:nvPr><p:ph type=\"sldNum\"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:fld><a:t>7</a:t></a:fld></a:p></p:txBody></p:sp>"
                           "<p:sp><p:nvSpPr><p:nvPr><p:ph type=\"ftr\"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Rahasia</a:t></a:r></a:p></p:txBody></p:sp>"
                           "<p:graphicFrame><a:tbl><a:tr><a:tc gridSpan=\"2\"><a:txBody><a:p><a:r><a:t>Gabung</a:t></a:r></a:p></a:txBody></a:tc>"
                           "<a:tc hMerge=\"1\"><a:txBody><a:p><a:r><a:t>Gabung</a:t></a:r></a:p></a:txBody></a:tc></a:tr>"
                           "<a:tr><a:tc><a:txBody><a:p><a:r><a:t>1</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>2</a:t></a:r></a:p></a:txBody></a:tc></a:tr>"
                           "</a:tbl></p:graphicFrame></p:spTree></p:cSld></p:sld>";
    const std::string smart = "<dgm:dataModel><dgm:ptLst><dgm:pt><dgm:t><a:bodyPr/><a:p><a:r><a:t>Rencana</a:t></a:r></a:p></dgm:t></dgm:pt>"
                              "<dgm:pt><dgm:t><a:p><a:r><a:t>Eksekusi</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>";
    const rdoc::Doc d = rdoc::read("a.pptx", deck({s1}, {{"ppt/slides/_rels/slide1.xml.rels",
        "<Relationships><Relationship Id=\"rId9\" Type=\"http://schemas/relationships/diagramData\" Target=\"../diagrams/data1.xml\"/></Relationships>"},
        {"ppt/diagrams/data1.xml", smart}}));
    check("pptx: title with an entity, no bullet, a line break", has(d, "Slide 1: Alur & Tahap", 1) && has(d, "tanpa bullet"), all(d));
    check("pptx: slide number and footer left out", all(d).find("Rahasia") == std::string::npos && all(d).find("7\n") == std::string::npos, all(d));
    check("pptx: a merged table cell once", has(d, "Gabung") && has(d, "1 | 2") && all(d).find("Gabung | Gabung") == std::string::npos, all(d));
    check("pptx: SmartArt text", has(d, "Rencana") && has(d, "Eksekusi"), all(d));
    {  // (mc:AlternateContent: the Choice read, its Fallback copy not)
      std::vector<rdoc::Para> body;
      std::string title;
      const std::string sp = "<p:sp><p:txBody><a:p><a:r><a:t>Rumus luas</a:t></a:r></a:p></p:txBody></p:sp>";
      pptx::slide_text("<p:sld><mc:AlternateContent><mc:Choice Requires=\"a14\">" + sp + "</mc:Choice><mc:Fallback>" + sp +
                       "</mc:Fallback></mc:AlternateContent></p:sld>", body, title, 1, false);
      check("pptx: AlternateContent read once", body.size() == 1 && body[0].text == "Rumus luas", std::to_string(body.size()));
    }
    check("pptx resolve", pptx::resolve("ppt/slides/", "../media/a.png") == "ppt/media/a.png" && pptx::resolve("ppt/slides/", "/ppt/x.xml") == "ppt/x.xml" &&
          pptx::resolve("ppt/", "../../../x") == "x");
    // hostile: missing slide parts, a relationship to itself, deep nesting, a binary .ppt, garbage
    std::string deep = "<p:sld>";
    for (int k = 0; k < 200000; ++k) deep += "<p:grpSp>";
    deep += "<p:sp><p:txBody><a:p><a:r><a:t>dalam</a:t></a:r></a:p></p:txBody></p:sp></p:sld>";
    const rdoc::Doc h = rdoc::read("h.pptx", deck({deep, "<p:sld/>"}, {{"ppt/slides/_rels/slide2.xml.rels",
        "<Relationships><Relationship Id=\"r\" Type=\"x/notesSlide\" Target=\"slide2.xml\"/></Relationships>"}}));
    check("pptx hostile: deep nesting, a slide's notes pointing at itself", has(h, "Slide 1: dalam", 1) && h.pages == 2, all(h));
    std::string many = "<p:presentation><p:sldIdLst>";
    for (int k = 0; k < 100000; ++k) many += "<p:sldId r:id=\"rId1\"/>";
    const std::string mz = doc::zip({{"ppt/presentation.xml", many + "</p:sldIdLst></p:presentation>"},
                                     {"ppt/_rels/presentation.xml.rels", "<Relationships><Relationship Id=\"rId1\" Target=\"slides/none.xml\"/></Relationships>"}});
    check("pptx hostile: 100,000 slide ids to a missing part, capped", rdoc::read("m.pptx", mz).pages <= 5000);
    check("pptx hostile: binary .ppt", rdoc::read("x.ppt", std::string("\xd0\xcf\x11\xe0\0\0\0\0", 8)).note.find("PPTX") != std::string::npos);
    check("pptx hostile: not a ZIP", !rdoc::read("x.pptx", "PK\x03\x04garbage").note.empty());
  }
  // CSV and TSV
  {
    const rdoc::Doc d = rdoc::read("a.csv", "Nama;Nilai;Catatan\r\nBudi;80;\"baik; rajin\"\r\nSiti;92,5;\"kata \"\"hebat\"\"\nbaris dua\"\r\n");
    check("CSV: semicolons, header names, quotes", has(d, "Nama: Budi | Nilai: 80 | Catatan: baik; rajin") && has(d, "Nama: Siti | Nilai: 92,5 | Catatan: kata \"hebat\" baris dua"), all(d));
    const rdoc::Doc n = rdoc::read("b.csv", "1,2,3\n4,,6\n");
    check("CSV: no header, an empty field kept", has(n, "1 | 2 | 3") && has(n, "4 |  | 6"), all(n));
    check("TSV", has(rdoc::read("c.tsv", "a\tb\nx\ty\n"), "a: x | b: y"));
    check("CSV: ANSI text", has(rdoc::read("d.csv", "kota\ncaf\xe9\n"), "kota: caf\xc3\xa9") || has(rdoc::read("d.csv", "kota\ncaf\xe9\n"), "caf\xc3\xa9"));
  }
  printf(fails ? "test_office: %d FAILED\n" : "test_office: all passed\n", fails);
  return fails ? 1 : 0;
}
