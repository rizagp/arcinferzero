// PowerPoint presentations with no library: uploaded .pptx / .pptm / .ppsx / .potx read as an rdoc::Doc for the web
// UI's file tools (tools/doc_reader.h calls read_pptx by the file's extension).
//
// The slides in their order (ppt/presentation.xml's slide list, ppt/_rels/presentation.xml.rels for each slide's
// part); each slide is a heading "Slide N: <title>" (its title placeholder, else its first line) on page N, so the
// outline lists the slides and read_file's pages are slides. Under it: the text of every shape in the slide's order
// (bullet points as "- " with two spaces per level, the footer / date / slide number placeholders left out), tables
// as rows (with a header row, "Column: value | ..." as spreadsheets, tools/xlsx.h), SmartArt text, a picture's alt
// text as "[image: ...]", and the speaker notes as "Notes: ...". Hidden slides are marked. Not read: text inside
// charts, embedded objects, the slide master's own text.
#pragma once
#include "xlsx.h"

namespace pptx {

// a relationship target against the folder of the part that names it ("ppt/slides/" + "../media/a.png" ->
// "ppt/media/a.png"; "/ppt/x.xml" from the package root)
inline std::string resolve(const std::string& dir, const std::string& target) {
  std::vector<std::string> parts;
  auto add = [&](const std::string& s) {
    size_t a = 0;
    while (a <= s.size()) {
      size_t e = s.find('/', a);
      if (e == std::string::npos) e = s.size();
      const std::string p = s.substr(a, e - a);
      if (p == "..") { if (!parts.empty()) parts.pop_back(); }
      else if (!p.empty() && p != ".") parts.push_back(p);
      a = e + 1;
    }
  };
  if (target.empty() || target[0] != '/') add(dir);
  add(target);
  std::string o;
  for (const std::string& p : parts) o += (o.empty() ? "" : "/") + p;
  return o;
}
struct Rel {
  std::string type, target;
};
// a part's relationships (id -> type, part), from <folder>/_rels/<name>.rels
inline std::map<std::string, Rel> rels_of(const std::string& zip, const std::string& part) {
  const size_t slash = part.rfind('/');
  const std::string dir = slash == std::string::npos ? "" : part.substr(0, slash + 1);
  std::string x;
  infl::zip_entry(zip, dir + "_rels/" + part.substr(slash == std::string::npos ? 0 : slash + 1) + ".rels", x, 4u << 20);
  std::map<std::string, Rel> m;
  xlsx::walk(x, [&](const std::string& n, const std::string& tag, size_t, size_t) {
    if (n == "Relationship" && xlsx::attr(tag, "TargetMode") != "External")
      m[xlsx::attr(tag, "Id")] = {xlsx::attr(tag, "Type"), resolve(dir, xlsx::attr(tag, "Target"))};
    return m.size() < 10000;
  });
  return m;
}

// a slide's (or notes page's, or SmartArt data's) text as paragraphs in body; its title placeholder's text in title
inline void slide_text(const std::string& x, std::vector<rdoc::Para>& body, std::string& title, int page, bool notes) {
  std::string ph, para, cell;
  bool in_sp = false, in_pic = false, in_para = false, bullet = false, in_tbl = false, skip_cell = false;
  int lvl = 0, row = 0, col = 0;
  xlsx::Grid grid;
  auto bullets = [&] { return !notes && in_sp && !ph.empty() && ph != "title" && ph != "ctrTitle" && ph != "subTitle"; };
  int fallback = 0;  // (inside mc:Fallback: a copy of the mc:Choice before it, for older readers: read once)
  xlsx::walk(x, [&](const std::string& n, const std::string& tag, size_t a, size_t b) {
    if (n == "Fallback") { if (!xlsx::self_closing(tag)) ++fallback; return true; }
    if (n == "/Fallback") { fallback = std::max(0, fallback - 1); return true; }
    if (fallback) return true;
    if (n == "sp") { in_sp = true; ph.clear(); }
    else if (n == "/sp") in_sp = false;
    else if (n == "ph" && in_sp) {
      ph = xlsx::attr(tag, "type");
      if (ph.empty()) ph = "body";  // (a placeholder without a type: the layout's content area)
    } else if (n == "pic") in_pic = !xlsx::self_closing(tag);
    else if (n == "/pic") in_pic = false;
    else if (n == "cNvPr" && in_pic) {
      const std::string alt = rdoc::tidy(xlsx::attr(tag, "descr"));
      if (!alt.empty()) body.push_back({"[image: " + alt + "]", page, 0});
    } else if (n == "tbl") {
      in_tbl = true;
      grid = xlsx::Grid{};
      row = 0;
    } else if (n == "tr" && in_tbl) { ++row; col = 0; }
    else if (n == "tc" && in_tbl) {
      ++col;
      cell.clear();
      skip_cell = xlsx::attr(tag, "hMerge") == "1" || xlsx::attr(tag, "vMerge") == "1";  // (covered by a merged cell)
    } else if (n == "/tc" && in_tbl) {
      const std::string t = rdoc::tidy(cell);
      double v;
      std::string f;
      if (!t.empty() && !skip_cell && grid.cells < 20000) {
        grid.rows[row][col] = {t, !(xlsx::date(t, v, f) || xlsx::number(t, v, f))};
        ++grid.cells;
      }
    } else if (n == "/tbl" && in_tbl) {
      in_tbl = false;
      rdoc::Doc d;
      if (!grid.rows.empty()) xlsx::grid_paras(grid, d);
      for (rdoc::Para& p : d.paras) body.push_back({p.text, page, 0});
    } else if (n == "p") {
      in_para = !xlsx::self_closing(tag);
      para.clear();
      lvl = 0;
      bullet = bullets();
    } else if (n == "pPr" && in_para) lvl = std::clamp(atoi(xlsx::attr(tag, "lvl").c_str()), 0, 8);
    else if (n == "buNone" && in_para) bullet = false;
    else if ((n == "buChar" || n == "buAutoNum") && in_para) bullet = true;
    else if (n == "br" && in_para) para += ' ';
    else if (n == "t" && in_para && !xlsx::self_closing(tag)) para += rdoc::xml_text(x.substr(a, b - a));
    else if (n == "/p" && in_para) {
      in_para = false;
      const std::string t = rdoc::tidy(para);
      if (t.empty()) return true;
      if (in_tbl) cell += (cell.empty() ? "" : " ") + t;
      else if (ph == "dt" || ph == "ftr" || ph == "sldNum" || ph == "hdr" || ph == "sldImg") {}
      else if (!notes && in_sp && (ph == "title" || ph == "ctrTitle")) title += (title.empty() ? "" : " ") + t;
      else body.push_back({(notes ? "Notes: " : "") + std::string((size_t)lvl * 2, ' ') + (bullet ? "- " : "") + t, page, 0});
    }
    return body.size() < 100000;
  });
}

}  // namespace pptx

namespace rdoc {

inline Doc read_pptx(const std::string& zip) {
  Doc d;
  if (zip.compare(0, 4, "\xd0\xcf\x11\xe0") == 0) {
    d.note = "a PowerPoint 97-2003 .ppt (binary): save it as PPTX to read it";
    return d;
  }
  std::string pres;
  if (!infl::zip_entry(zip, "ppt/presentation.xml", pres, 16u << 20)) {
    d.note = "not a readable PPTX (no ppt/presentation.xml)";
    return d;
  }
  const auto prels = pptx::rels_of(zip, "ppt/presentation.xml");
  std::vector<std::string> slides;
  xlsx::walk(pres, [&](const std::string& n, const std::string& tag, size_t, size_t) {
    if (n == "sldId") {
      auto r = prels.find(xlsx::attr(tag, "r:id"));
      if (r != prels.end()) slides.push_back(r->second.target);
    }
    return slides.size() < 5000;
  });
  size_t chars = 0, inflated = 0, unread = 0;
  for (size_t k = 0; k < slides.size(); ++k) {
    const int page = (int)k + 1;
    if (inflated > (512u << 20)) { unread = slides.size() - k; break; }  // (all parts together: a bounded read)
    std::string x;
    infl::zip_entry(zip, slides[k], x, 16u << 20);
    inflated += x.size();
    bool hidden = false;
    xlsx::walk(x, [&](const std::string& n, const std::string& tag, size_t, size_t) {
      if (n == "sld") hidden = xlsx::attr(tag, "show") == "0" || xlsx::attr(tag, "show") == "false";
      return n != "sld";
    });
    std::vector<Para> body, smart, notes;
    std::string title, ignored;
    pptx::slide_text(x, body, title, page, false);
    auto ends = [](const std::string& s, const char* t) { const size_t n = strlen(t); return s.size() >= n && s.compare(s.size() - n, n, t) == 0; };
    for (const auto& [id, rel] : pptx::rels_of(zip, slides[k])) {
      std::string part;
      const bool is_notes = ends(rel.type, "/notesSlide");
      if ((is_notes || ends(rel.type, "/diagramData")) && inflated <= (512u << 20) && infl::zip_entry(zip, rel.target, part, 16u << 20)) {
        inflated += part.size();
        pptx::slide_text(part, is_notes ? notes : smart, ignored, page, is_notes);
      }
    }
    body.insert(body.end(), smart.begin(), smart.end());
    if (title.empty())  // (no title placeholder: its first line)
      for (size_t i = 0; i < body.size(); ++i)
        if (body[i].text.rfind("[image: ", 0) != 0) {
          title = Index::cut_words(body[i].text, 80);
          if (title == body[i].text) body.erase(body.begin() + (long long)i);
          break;
        }
    body.insert(body.end(), notes.begin(), notes.end());
    d.paras.push_back({"Slide " + std::to_string(page) + (title.empty() ? "" : ": " + title) + (hidden ? " (hidden)" : ""), page, 1});
    for (Para& p : body) {
      chars += p.text.size();
      d.paras.push_back(std::move(p));
    }
    if (chars > (64u << 20)) {
      d.note = "large presentation: read up to slide " + std::to_string(page) + " of " + std::to_string(slides.size());
      break;
    }
  }
  d.pages = (int)slides.size();
  if (unread) d.note = "large presentation: its parts unpack to more than 512 MB; " + std::to_string(unread) + " of " +
                       std::to_string(slides.size()) + " slides not read";
  if (slides.empty()) d.note = "the presentation has no slides";
  return d;
}

}  // namespace rdoc
