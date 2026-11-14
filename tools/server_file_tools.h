// A part of tools/server.cpp, included at the place it was cut from (the translation unit is the same text), not
// a header of its own: the file tools the web UI's model calls (/v1/tools/*) and the files API (/v1/files, /v1/chats).

// The web UI's file tools for the model (it calls them as functions, the page runs them here):
//   create_file {filename, format: docx | pdf | doc | rtf | xlsx,
//                    made from Markdown content (tools/doc_writer.h; xlsx: each table a sheet, tools/xlsx.h;
//                    number_format auto | id | us), or any text extension (md, html, py...: saved as written), title,
//                    subtitle, author, date, page, orientation, margins_mm, font, font_size, line_spacing,
//                    heading_color, accent_color, text_color, header, footer, page_numbers, chat_id}: writes the file
//                    into the chat's folder (and its source next to it), returns it as a file
//   list_files {kind, name_contains, within_hours, chat_id, limit}: the saved files, newest first, with their metadata
//   read_file {file_id (or its name), section, pages, offset, max_chars, chat_id}: a document's text (DOCX, PDF, XLSX
//                    and text files, tools/doc_reader.h) in parts (a fifth of the loaded model's context at most, the rest by
//                    offset: files::read_budget); a long one without section / pages / offset: its outline;
//                    a document the model wrote: its source (to make a corrected version); images, audio: metadata.
//                    A PDF's scanned pages (images without a text layer) read by pages: the loaded model transcribes
//                    them from their images first, up to 4 a call, when it has an image encoder (FileVision)
//   digest_file {file_id, focus, chat_id}: a document longer than one read, read part by part by the loaded model
//                    (its own requests), notes of every part saved as "<name> (notes).md" and returned
//   search_file {query, file_id, chat_id, top_k}: the passages of a document (or of every document of the chat)
//                    that fit the query best (BM25 over chunks of ~2,400 characters), with their section and pages
//   read_file / search_file also take "chat:<id>" (another chat: a section per exchange) and "transcript:<id>"
//                    (a recording or meeting of the Whisper library: its summary, then 5-minute sections) as file_id
//   search_chats {query, limit, chat_id}: the other chats; no query: newest first (limit 1-100, 20); a query: their
//                    exchanges that fit it best (BM25, limit 1-10, 5)
//   run_command {command, chat_id, timeout_s}: one PowerShell command in the chat's folder (tools/server_terminal.h),
//                    sent by the page only after the user allowed it; refused (403) to any computer but this one.
//                    session: true: in the chat's terminal (folder and environment kept for the next command; the
//                    reply says cwd and venv); reset: true: that terminal closed (with no command: only that)
// ctx: the loaded model's context in tokens per conversation (0: not known): a read or a search gives at most what
// files::read_budget allows, the rest by offset
// Scanned pages are transcribed here, by the server, not attached to the model's next turn as images: the text is
// kept per file and page (tools/server_files.h), so search_file finds it and a later chat reads it without the
// vision model; in the conversation a page costs its text (~300-800 tokens) instead of an image's ~1,000 tokens on
// every later turn; and the web UI needs no image tool results, which the chat templates do not all take. The cost:
// the transcription is a conversation of its own, so with one slot the chat's prompt is processed again at its next
// turn (--slots 2 or the session cache keep it).
struct FileVision {
  std::function<std::string()> model;  // the loaded model's name when it has an image encoder, else ""
  std::function<std::string(const std::string& bytes, const std::string& mime)> see;  // an image -> its text
  // a prompt of words -> the loaded model's answer (no thinking, up to max_tokens; digest_file)
  std::function<std::string(const std::string& prompt, int max_tokens)> ask;
};
static constexpr const char* kTranscribe =
    "Transcribe all the text on this scanned page exactly as written, in reading order and in its own language. Use "
    "Markdown: # for headings, - for list items, | cells | for table rows. Write only the page's text, without comments; "
    "a page without text: (no text)";
static void file_tools(const HttpRequest& rq, HttpResponse& rs, bool terminal, int ctx, const FileVision& vis) {
  auto& st = files::store();
  auto reply = [&](int code, const Json& j) { rs.send(code, "application/json", j.dump()); };
  auto fail = [&](int code, const std::string& m) {
    Json e = Json::object(), r = Json::object();
    e.set("message", m);
    r.set("error", e);
    reply(code, r);
  };
  Json a;
  if (!Json::try_parse(rq.body, a) || !a.is_object()) return fail(400, "expected a JSON object");
  auto when = [](int64_t t) {
    char b[40] = "";
    const std::time_t ct = (std::time_t)t;
    std::tm tm{};
    if (plat::local_time(ct, tm)) strftime(b, sizeof b, "%Y-%m-%d %H:%M", &tm);
    return std::string(b);
  };
  try {
    if (rq.path == "/v1/tools/create_file") {
      const std::string content = a["content"].str();
      if (content.empty()) return fail(400, "content required");
      // docx, pdf, doc, rtf, xlsx are made from Markdown; md and every other text format (html, svg, css, js, py,
      // csv, json, txt...) are saved as written. No format: the file name's extension, else docx
      std::string fmt = files::lower(a["format"].str(files::ext_of(a["filename"].str()).empty() ? "docx" : files::ext_of(a["filename"].str())));
      if (!fmt.empty() && fmt[0] == '.') fmt.erase(0, 1);
      bool fmt_ok = !fmt.empty() && fmt.size() <= 10;
      for (unsigned char c : fmt) fmt_ok = fmt_ok && c < 128 && isalnum(c);
      if (!fmt_ok) return fail(400, "format: an extension such as docx, pdf, xlsx, md, html, svg, csv, json, py, txt");
      // (binary formats cannot be written from text: images, audio, video, archives, programs, other Office files)
      static const std::set<std::string> binary = {"pptx", "ppt", "xls", "xlsm", "odt", "ods", "odp", "zip", "7z", "rar",
          "gz", "tar", "exe", "dll", "msi", "bin", "gguf", "ico", "tif", "tiff", "heic", "woff", "woff2", "ttf", "otf"};
      if (files::kind_of("x." + fmt) != "document" || binary.count(fmt))
        return fail(400, "format " + fmt + " is binary and cannot be written from text; for an image write svg, for a "
                         "presentation docx or pdf, for anything else a script with run_command if the terminal is offered");
      doc::Options o;
      o.title = a["title"].str();
      o.subtitle = a["subtitle"].str();
      o.author = a["author"].str();
      o.date = a["date"].str();
      o.page = a["page"].str("A4");
      o.landscape = files::lower(a["orientation"].str()) == "landscape";
      if (a["font"].is_string() && !a["font"].str().empty()) o.font = a["font"].str();
      if (a["font_size"].is_number()) o.font_size = std::clamp(a["font_size"].as_number(), 6.0, 36.0);
      if (a["line_spacing"].is_number()) o.line_spacing = std::clamp(a["line_spacing"].as_number(), 0.8, 3.0);
      if (a["margins_mm"].is_number()) for (double& m : o.margin_mm) m = std::clamp(a["margins_mm"].as_number(), 5.0, 60.0);
      else if (a["margins_mm"].is_array())
        for (size_t k = 0; k < 4 && k < a["margins_mm"].items().size(); ++k) o.margin_mm[k] = std::clamp(a["margins_mm"].items()[k].as_number(20), 5.0, 60.0);
      for (auto [key, dst] : {std::pair<const char*, std::string*>{"heading_color", &o.heading_color}, {"accent_color", &o.accent_color}, {"text_color", &o.text_color}})
        if (const std::string c = doc::hex_color(a[key].str()); !c.empty()) *dst = c;
      o.header = a["header"].str();
      o.footer = a["footer"].str();
      if (a["page_numbers"].is_bool()) o.page_numbers = a["page_numbers"].as_bool();
      // (an empty filename too: the title, else "document"; before, "" made "file.docx")
      const std::string fname = doc::trim(a["filename"].str());
      std::string name = files::safe_name(!fname.empty() ? fname : o.title.empty() ? "document" : o.title);
      if (files::ext_of(name) != fmt) name += "." + fmt;
      std::string data;
      if (fmt == "docx") data = doc::docx(content, o);
      else if (fmt == "pdf") data = doc::pdf(content, o);
      else if (fmt == "doc" || fmt == "rtf") data = doc::rtf(content, o);
      else if (fmt == "xlsx") data = xlsx::write(content, o, a["number_format"].str("auto"));
      // (md: the title as its heading, unless the content starts with a heading of its own: else it came twice)
      else if (fmt == "md") data = (o.title.empty() || doc::trim(content).rfind("# ", 0) == 0 ? "" : "# " + o.title + "\n\n") + content;
      else data = content;
      const std::string chat = files::valid_id(a["chat_id"].str()) ? a["chat_id"].str() : "";
      const int64_t before = (int64_t)std::time(nullptr);
      const files::Info i = st.put(name, data, chat);
      // (the same bytes saved before: put gives that file back, under its own name; its source stays the one that
      //  made it, and the card says so, instead of this request's source and name over it)
      const bool existed = i.created < before;
      if (!existed) st.put_source(i.id, chat, rq.body);
      Json r = files::info_json(i);
      r.set("file_id", i.id);
      r.set("note", existed ? "these exact bytes were already saved as " + i.name + ": that file is shown to the user (the "
                              "same file card); to make a different file, change its content"
                            // (it said "to change it ... call create_file again: that makes a new version", and a model
                            //  writing a meeting's minutes went on making versions of its own accord: six files for
                            //  one request, 2026-11-04. Now: done, unless the user asks for a change)
                            : "created: the file is done and the user sees it as a file card in your answer (download / "
                              "open). Do not make it again or another version unless the user asks for a change; then "
                              "read its source with read_file and call create_file again.");
      return reply(200, r);
    }
    // Word templates of the user (%USERPROFILE%\.inferzero\templates\*.docx: a company's minutes forms, with their
    // letterhead and signature block), tools/doc_template.h. minutes_templates {}: each one's name and what it asks
    // for (its first table's labels, its other tables' headings and columns). minutes_docx {template, data,
    // filename, chat_id}: that template filled with data ({"fields": {...}, "tables": [[[cells]...]...]}), saved
    // as a file as create_file saves one.
    if (rq.path == "/v1/tools/minutes_templates" || rq.path == "/v1/tools/minutes_docx") {
      namespace fs = std::filesystem;
      const fs::path dir = fs::path(app_dir()) / "templates";
      auto read = [](const fs::path& p) {
        std::ifstream f(p, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), {});
      };
      std::error_code ec;
      if (rq.path == "/v1/tools/minutes_templates") {
        Json list = Json::array();
        for (const auto& e : fs::directory_iterator(dir, ec)) {
          if (!e.is_regular_file() || files::lower(e.path().extension().string()) != ".docx") continue;
          const std::string n = e.path().filename().string();
          if (n.rfind("~$", 0) == 0) continue;  // (Word's lock file of an open document)
          std::string bad;
          const Json s = doct::structure(read(e.path()), bad);
          if (!bad.empty()) continue;
          Json t = s;
          t.set("name", e.path().stem().string());
          list.push(t);
        }
        Json r = Json::object();
        r.set("templates", list);
        r.set("folder", dir.string());
        return reply(200, r);
      }
      const std::string tn = a["template"].str();
      if (tn.empty() || tn.find_first_of("/\\:") != std::string::npos || tn.find("..") != std::string::npos)
        return fail(400, "template: a template's name (minutes_templates lists them)");
      const fs::path tp = dir / (tn + ".docx");
      if (!fs::is_regular_file(tp, ec)) return fail(404, "no template " + tn + " in " + dir.string());
      std::string bad;
      const std::string data = doct::fill(read(tp), a["data"], bad);
      if (data.empty()) return fail(400, "the template " + tn + ": " + bad);
      std::string name = files::safe_name(doc::trim(a["filename"].str()).empty() ? tn : doc::trim(a["filename"].str()));
      if (files::ext_of(name) != "docx") name += ".docx";
      const std::string chat = files::valid_id(a["chat_id"].str()) ? a["chat_id"].str() : "";
      const files::Info i = st.put(name, data, chat);
      Json r = files::info_json(i);
      r.set("file_id", i.id);
      r.set("note", "created from the template " + tn);
      return reply(200, r);
    }
    if (rq.path == "/v1/tools/list_files" && a["scope"].str() == "project") {  // (the chat's project folder)
      proj::Project p;
      if (!rq.local_peer) return fail(403, "a project's files are only for this computer");
      if (!memo::valid_project(a["project_id"].str()) || !proj::find(a["project_id"].str(), p))
        return fail(400, "this chat is not in a project (scope \"project\" lists a project's folder)");
      const std::string kind = files::lower(a["kind"].str()), q = files::lower(a["name_contains"].str());
      const double hours = a["within_hours"].as_number(0);
      const int limit = (int)std::clamp(a["limit"].as_number(50), 1.0, 300.0);
      bool more = false;
      const int64_t now = (int64_t)std::time(nullptr);
      Json d = Json::array();
      for (const proj::Entry& e : proj::walk(p, 20000, more)) {
        if (!kind.empty() && files::kind_of(e.rel) != kind) continue;
        if (!q.empty() && files::lower(e.rel).find(q) == std::string::npos) continue;
        if (hours > 0 && now - e.modified > hours * 3600) continue;
        if ((int)d.size() >= limit) {
          more = true;
          break;
        }
        Json j = Json::object();
        j.set("file_id", "project:" + p.id + "/" + e.rel);
        j.set("path", e.rel);
        j.set("kind", files::kind_of(e.rel));
        j.set("bytes", (double)e.bytes);
        j.set("modified", when(e.modified));
        d.push(j);
      }
      Json r = Json::object();
      r.set("project", p.name);
      r.set("folder", p.folder);
      r.set("files", d);
      r.set("count", (double)d.size());
      r.set("note", std::string(more ? "more files than these (newest first): narrow with name_contains or kind. " : "") +
                        "read one with read_file {\"file_id\": \"project:...\"}; files you write with create_file go to the "
                        "chat's own folder: to put one into the project, the user copies it, or run_command (they allow it)");
      return reply(200, r);
    }
    if (rq.path == "/v1/tools/list_files") {
      const std::string kind = files::lower(a["kind"].str()), q = files::lower(a["name_contains"].str()), chat = a["chat_id"].str();
      const double hours = a["within_hours"].as_number(0);
      // (20 by default, and no more than one read of the model's context: 50 entries of ~170 characters stayed in
      //  every later prompt of a small context)
      const int limit = (int)std::clamp(a["limit"].as_number(20), 1.0, 200.0);
      const auto uses = st.uses();
      const int64_t now = (int64_t)std::time(nullptr);
      Json d = Json::array();
      size_t given = 0;
      bool more = false;
      for (const files::Info& i : st.list()) {
        if (!kind.empty() && i.kind != kind) continue;
        if (!q.empty() && files::lower(i.name).find(q) == std::string::npos) continue;
        if (hours > 0 && now - i.created > hours * 3600) continue;
        auto u = uses.find(i.id);
        const auto held = st.holders(i.id);
        if (!chat.empty() && std::find(held.begin(), held.end(), chat) == held.end() &&
            (u == uses.end() || std::find(u->second.begin(), u->second.end(), chat) == u->second.end()))
          continue;
        Json j = Json::object();
        j.set("file_id", i.id);
        j.set("filename", i.name);
        j.set("kind", i.kind);
        j.set("bytes", (double)i.bytes);
        j.set("uploaded", when(i.created));
        if (i.kind == "image" || i.kind == "audio" || i.kind == "video") {  // (its size or length: an image's header)
          std::string head(i.kind == "image" ? 65536 : 0, '\0');
          std::ifstream hf(std::filesystem::path(i.path), std::ios::binary);
          hf.read(head.data(), (std::streamsize)head.size());
          head.resize((size_t)hf.gcount());
          const std::string x = files::extent_of(i, head);
          if (!x.empty()) j.set(i.kind == "image" ? "size" : "length", x.substr(2));
        }
        std::string src;
        j.set("written_by_model", st.source(i.id, src));
        j.set("chats", (double)std::max(held.size(), u == uses.end() ? size_t(0) : u->second.size()));
        const size_t sz = j.dump().size();
        if ((int)d.size() >= limit || (d.size() && given + sz > files::read_budget(ctx))) { more = true; break; }
        given += sz;
        d.push(j);
      }
      Json r = Json::object();
      r.set("files", d);
      r.set("count", (double)d.size());
      if (more) r.set("note", "more files than these (newest first): narrow with name_contains, kind or within_hours, or a larger limit");
      return reply(200, r);
    }
    // the call's file: by its id, or by its name (models often pass the name they see: the newest file of that
    // name, this chat's first)
    // a project's file ("project:<id>/<path>", tools/server_projects.h): its Info from the file itself; its stamp (size
    // and time) keys the document cache, so a file that changed is read again (files::doc_index)
    // Only the chat's own project (the call's project_id, set by the web UI) and on this computer: a document or page
    // the model read cannot send it into another project's files
    std::string proj_err;
    proj::Project chat_proj;  // (the call's project, read from projects.json once a call)
    bool chat_proj_read = false;
    auto project_file = [&](const std::string& fid, files::Info& i) {
      std::filesystem::path f;
      std::string id, rel;
      if (!proj::split_id(fid, id, rel)) return false;
      if (!rq.local_peer) {
        proj_err = "a project's files are only for this computer";
        return false;
      }
      if (id != a["project_id"].str()) {
        proj_err = "project:" + id + " is not this chat's project: only the files of the chat's own project can be read";
        return false;
      }
      if (!chat_proj_read) {
        chat_proj_read = true;
        if (!memo::valid_project(id) || !proj::find(id, chat_proj)) chat_proj.id.clear();
      }
      if (chat_proj.id != id) {
        proj_err = "no project " + id;
        return false;
      }
      if (!proj::resolve_in(chat_proj, rel, f, proj_err)) return false;
      std::error_code ec;
      i = files::Info();
      i.path = f.string();
      i.name = rel;
      i.kind = files::kind_of(rel);
      i.bytes = std::filesystem::file_size(f, ec);
      const auto t = std::filesystem::last_write_time(f, ec);
#ifdef _WIN32
      i.created = (int64_t)std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count() - 11644473600LL;
#else
      // (unix seconds through clock_cast, as server_files.h: libstdc++'s file clock does not count from 1601 as MSVC's,
      //  and the subtraction above gave dates around the year 1453)
      i.created = (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::clock_cast<std::chrono::system_clock>(t).time_since_epoch()).count();
#endif
      i.id = fid;
      i.stamp = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count();
      return true;
    };
    auto find_file = [&](files::Info& i) {
      std::string id = a["file_id"].str();
      if (id.rfind("project:", 0) == 0) return project_file(id, i);
      if (id.rfind("file-", 0) == 0) id = id.substr(5);
      if (id.rfind("/files/", 0) == 0) id = id.substr(7);
      if (files::valid_id(id, 16) && st.get(id, i)) return true;
      const std::string want = files::lower(a["file_id"].str()), chat = a["chat_id"].str();
      bool found = false;
      for (const files::Info& x : st.list()) {
        if (files::lower(x.name) != want) continue;
        const auto held = st.holders(x.id);
        const bool here = std::find(held.begin(), held.end(), chat) != held.end();
        if (!found || here) { i = x; found = true; }
        if (here) break;
      }
      return found;
    };
    if (rq.path == "/v1/tools/search_file") {
      const std::string query = a["query"].str();
      if (query.empty()) return fail(400, "query required");
      const size_t top = (size_t)std::clamp(a["top_k"].as_number(5), 1.0, 10.0);
      std::vector<files::Info> targets;
      std::vector<std::shared_ptr<const rdoc::Index>> ixs;
      if (files::is_virtual(a["file_id"].str())) {  // (another chat, or a transcript)
        files::VirtualDoc v;
        if (!files::virtual_doc(a["file_id"].str(), v)) return fail(404, "no " + a["file_id"].str() + " (search_chats without a query lists the chats)");
        files::Info i;
        i.id = a["file_id"].str();
        i.name = v.title;
        targets.push_back(i);
        ixs.push_back(v.ix);
      } else if (!a["file_id"].str().empty()) {
        files::Info i;
        if (!find_file(i)) return fail(404, !proj_err.empty() ? proj_err : "no file " + a["file_id"].str() + " (list_files shows the saved files)");
        targets.push_back(i);
      } else if (files::valid_id(a["chat_id"].str()) || memo::valid_project(a["project_id"].str())) {  // (every document of this chat, and of its project)
        proj::Project pr;
        if (rq.local_peer && memo::valid_project(a["project_id"].str()) && proj::find(a["project_id"].str(), pr)) {
          // (its documents and sources, newest first: at most 60 files and 32 MB a search, each read again when it
          //  changed; programs, archives and models are left out)
          static const std::set<std::string> text = {"docx", "pdf", "xlsx", "xlsm", "pptx", "rtf", "txt", "md", "csv", "tsv",
              "json", "html", "htm", "xml", "py", "js", "ts", "tsx", "jsx", "c", "cc", "cpp", "h", "hpp", "cs", "java", "go",
              "rs", "ps1", "sh", "bat", "cmd", "yml", "yaml", "toml", "ini", "cfg", "sql", "css", "cl", "lua", "rb", "php"};
          bool more = false;
          size_t taken = 0;
          uint64_t bytes = 0;
          for (const proj::Entry& e : proj::walk(pr, 20000, more)) {
            if (!text.count(files::ext_of(e.rel)) || bytes + e.bytes > (32ull << 20)) continue;
            files::Info i;
            if (!project_file("project:" + pr.id + "/" + e.rel, i)) continue;
            targets.push_back(i);
            bytes += e.bytes;
            if (++taken >= 60) break;
          }
        }
        const std::string chat = a["chat_id"].str();
        const auto uses = st.uses();
        for (const files::Info& i : st.list()) {
          if (i.kind != "document") continue;
          const auto held = st.holders(i.id);
          auto u = uses.find(i.id);
          if (std::find(held.begin(), held.end(), chat) != held.end() ||
              (u != uses.end() && std::find(u->second.begin(), u->second.end(), chat) != u->second.end()))
            targets.push_back(i);
        }
      } else return fail(400, "file_id (or chat_id / project_id: every document of the chat / project) required");
      struct Found { double score; size_t file, chunk; };
      std::vector<Found> all;
      for (size_t f = 0; f < targets.size(); ++f) {
        if (ixs.size() <= f) ixs.push_back(files::doc_index(targets[f]));
        if (!ixs[f]) continue;
        for (const auto& h : ixs[f]->search(query, top)) all.push_back({h.score, f, h.chunk});
      }
      std::sort(all.begin(), all.end(), [](const Found& x, const Found& y) { return x.score > y.score; });
      if (all.size() > top) all.resize(top);
      Json res = Json::array();
      size_t given = 0, left_out = 0;  // (the passages' text within the read budget; the best one always)
      for (const Found& h : all) {
        const rdoc::Index& ix = *ixs[h.file];
        const rdoc::Chunk& c = ix.chunks[h.chunk];
        if (res.size() && given + c.text.size() > files::read_budget(ctx)) { ++left_out; continue; }
        given += c.text.size();
        Json j = Json::object();
        j.set("file_id", targets[h.file].id);
        j.set("filename", targets[h.file].name);
        j.set("section", (double)(c.section + 1));
        j.set("section_title", ix.outline[(size_t)c.section].title);
        if (ix.doc.pages > 1) j.set("pages", c.page_from == c.page_to ? std::to_string(c.page_from) : std::to_string(c.page_from) + "-" + std::to_string(c.page_to));
        char sc[16];
        snprintf(sc, sizeof sc, "%.1f", h.score);
        j.set("score", std::string(sc));
        j.set("text", c.text);
        res.push(j);
      }
      Json r = Json::object();
      r.set("results", res);
      r.set("files_searched", (double)targets.size());
      std::string unread;  // (scanned pages no vision model has read yet: their text is not searched)
      for (size_t f = 0; f < targets.size(); ++f)
        if (ixs[f] && !ixs[f]->doc.unread.empty())
          unread += (unread.empty() ? "" : "; ") + targets[f].name + ": pages " + rdoc::pdfimg::ranges(ixs[f]->doc.unread);
      if (!unread.empty())
        r.set("not_searched", "scanned pages not transcribed yet (" + unread + "): read_file with pages has a vision model read them");
      // (nothing to search is not "no passage has these words": that sent the model off retrying with other words)
      const size_t searchable = (size_t)std::count_if(ixs.begin(), ixs.end(), [](const auto& x) { return x != nullptr; });
      if (targets.empty())
        r.set("note", "this chat has no documents to search: list_files shows the saved files (then search_file with file_id)");
      else if (searchable == 0)
        r.set("note", targets.size() == 1 ? targets[0].name + " has no text to search (" + targets[0].kind + "): read_file gives what it holds"
                                          : "none of these files has text to search");
      else if (res.size() == 0)
        r.set("note", "no passage has these words: try other words (the document's own terms, synonyms), or read the outline (read_file)");
      else if (left_out)
        r.set("note", std::to_string(left_out) + " more passages left out to keep within the model's context: search with more "
                      "precise words, or read a result's section (read_file)");
      return reply(200, r);
    }
    // the other chats (the web UI's, one reads another): listed newest first (a query: in their text), or searched
    // by their exchanges (BM25); each is read with read_file {"file_id": "chat:<id>"} (an outline per exchange)
    // (one tool for both since 2026-10-28, list_chats before: no query lists them, a query searches their exchanges)
    if (rq.path == "/v1/tools/search_chats") {
      const std::string q = a["query"].str(), self = a["chat_id"].str();
      struct C { std::string id, title, model, snippet; int64_t at; };
      std::vector<C> cs;
      for (const auto& [cid, text] : st.chats_all()) {
        if (cid == self) continue;
        Json c;
        if (!Json::try_parse(text, c)) continue;
        cs.push_back({cid, c["title"].str("chat"), c["model"].str(), c["snippet"].str(), (int64_t)(c["at"].as_number(0) / 1000)});
      }
      std::sort(cs.begin(), cs.end(), [](const C& x, const C& y) { return x.at > y.at; });
      if (doc::trim(q).empty()) {
        const size_t limit = (size_t)std::clamp(a["limit"].as_number(20), 1.0, 100.0);
        Json d = Json::array();
        size_t given = 0;  // (no more than one read of the model's context, as list_files)
        for (const C& c : cs) {
          if (d.size() >= limit) break;
          files::VirtualDoc v;
          Json j = Json::object();
          j.set("file_id", "chat:" + c.id);
          j.set("title", c.title);
          j.set("updated", when(c.at));
          if (files::virtual_doc("chat:" + c.id, v)) j.set("exchanges", (double)v.parts);
          if (!c.model.empty()) j.set("model", c.model);
          if (!c.snippet.empty()) j.set("snippet", c.snippet);
          const size_t sz = j.dump().size();
          if (d.size() && given + sz > files::read_budget(ctx)) break;
          given += sz;
          d.push(j);
        }
        Json r = Json::object();
        r.set("chats", d);
        r.set("count", (double)d.size());
        r.set("note", "read one with read_file {\"file_id\": \"chat:<id>\"} (long ones: their outline, one section per exchange), "
                      "or find passages in all of them with search_chats and a query");
        return reply(200, r);
      }
      const size_t top = (size_t)std::clamp(a["limit"].as_number(5), 1.0, 10.0);
      struct Hit { double score; size_t chat, chunk; };
      std::vector<Hit> hits;
      std::vector<std::shared_ptr<const rdoc::Index>> ixs(cs.size());
      for (size_t k = 0; k < cs.size(); ++k) {
        files::VirtualDoc v;
        if (!files::virtual_doc("chat:" + cs[k].id, v)) continue;
        ixs[k] = v.ix;
        for (const auto& h : v.ix->search(q, top)) hits.push_back({h.score, k, h.chunk});
      }
      std::sort(hits.begin(), hits.end(), [](const Hit& x, const Hit& y) { return x.score > y.score; });
      if (hits.size() > top) hits.resize(top);
      Json res = Json::array();
      size_t given = 0;
      for (const Hit& h : hits) {
        const rdoc::Chunk& c = ixs[h.chat]->chunks[h.chunk];
        if (res.size() && given + c.text.size() > files::read_budget(ctx)) continue;
        given += c.text.size();
        Json j = Json::object();
        j.set("file_id", "chat:" + cs[h.chat].id);
        j.set("title", cs[h.chat].title);
        j.set("section", (double)(c.section + 1));
        j.set("exchange", ixs[h.chat]->outline[(size_t)c.section].title);
        char sc[16];
        snprintf(sc, sizeof sc, "%.1f", h.score);
        j.set("score", std::string(sc));
        j.set("text", c.text);
        res.push(j);
      }
      Json r = Json::object();
      r.set("results", res);
      r.set("chats_searched", (double)cs.size());
      if (res.size() == 0) r.set("note", "no exchange has these words: try other words, or search_chats without a query (the chats, newest first)");
      return reply(200, r);
    }
    // digest_file {file_id, focus, chat_id}: a document longer than the model's context, read for it part by part. The
    // model's tool rounds ran out on such a document read in parts (2026-10-31), and a chat's compaction keeps no read
    // position; so the server takes the turns: each part (as much as the context takes, at paragraph ends) goes to the
    // loaded model as a request of its own, "notes of this part" (focus: what the user asked); the notes go into
    // "<name> (notes).md" in the chat's folder (a later call with the same file and focus reads them from there, an
    // unfinished one goes on from its last part), and come back as the result, combined again while they are longer
    // than one read. A document that fits one read comes whole, as read_file gives it.
    if (rq.path == "/v1/tools/digest_file") {
      files::Info i;
      if (files::is_virtual(a["file_id"].str())) return fail(400, "digest_file reads saved files; a chat or transcript: read_file");
      if (!find_file(i)) return fail(404, !proj_err.empty() ? proj_err : "no file " + a["file_id"].str() + " (list_files shows the saved files)");
      auto ix = files::doc_index(i);
      if (!ix) return fail(500, "cannot read the file");
      if (ix->doc.paras.empty()) return fail(400, ix->doc.note.empty() ? "no text in the file" : ix->doc.note);
      const size_t n = ix->doc.paras.size(), budget = files::read_budget(ctx);
      const std::string focus = a["focus"].str().substr(0, 600);
      Json r = Json::object();
      r.set("file_id", i.id);
      r.set("filename", i.name);
      const std::string all = ix->text(0, n);
      r.set("total_tokens", (double)files::tokens_of(all.size()));
      if (ctx > 0 && all.size() <= budget) {
        r.set("content", all);
        r.set("note", "the whole document (it fits your context): no notes needed");
        return reply(200, r);
      }
      if (!vis.ask) return fail(503, "no model is loaded to read the parts");
      // the parts: up to ~45 % of the context each (characters ~3.5 a token), ending before a section's heading when
      // one is in the part's second half (a section whole in one part), else before a page's heading, else at a
      // paragraph's end; a part that starts inside a section is told which one it goes on with
      const size_t part = std::clamp<size_t>(ctx > 0 ? (size_t)ctx * 3 / 2 : 24000, 4000, 60000);
      auto page_head = [&](size_t k) {
        int x = 0, y = 0;
        return ix->doc.paras[k].level > 0 && sscanf(ix->doc.paras[k].text.c_str(), "Page %d / %d", &x, &y) == 2;
      };
      // a source file has no headings: a paragraph that starts at column 0 (not a closing brace or a comment's end)
      // begins a top-level definition, Python's def / class, a C / JS function, so a function stays in one part
      static const std::set<std::string> kCode = {"py", "js", "mjs", "ts", "tsx", "jsx", "c", "cc", "cpp", "cxx", "h", "hpp",
                                                  "cs", "java", "go", "rs", "rb", "php", "kt", "swift", "ps1", "psm1", "sh",
                                                  "bash", "lua", "sql", "scala", "dart", "vb", "r", "m", "pl"};
      const bool code = kCode.count(rdoc::lower_ext(i.name)) > 0;
      auto top_level = [&](size_t k) {
        const std::string& t = ix->doc.paras[k].text;
        return code && !t.empty() && !isspace((unsigned char)t[0]) && t[0] != '}' && t[0] != ')' && t[0] != ']' &&
               t.rfind("*/", 0) != 0 && t.rfind("end", 0) != 0;
      };
      auto section_head = [&](size_t k) { return (ix->doc.paras[k].level > 0 && !page_head(k)) || top_level(k); };
      std::vector<std::pair<size_t, size_t>> parts;
      for (size_t from = 0; from < n;) {
        size_t k = from, sz = 0;
        while (k < n && (k == from || sz + ix->doc.paras[k].text.size() + 2 <= part)) sz += ix->doc.paras[k++].text.size() + 2;
        size_t cut = k;
        if (k < n) {
          const size_t half = from + (k - from) / 2 + 1;
          for (bool sec : {true, false}) {
            size_t j = k;
            while (j > half && !(sec ? section_head(j) : page_head(j))) --j;
            if (j > half) { cut = j; break; }
          }
        }
        parts.push_back({from, cut});
        from = cut;
      }
      // each part's text; one paragraph longer than a part (a long function without a blank line, a big table) is
      // cut at its lines, so no part's prompt goes past the context
      struct Part { size_t first, last; std::string text; };
      std::vector<Part> pts;
      for (const auto& [f, l] : parts) {
        std::string t = ix->text(f, l);
        if (t.size() <= part + part / 4) { pts.push_back({f, l, std::move(t)}); continue; }
        for (size_t at = 0; at < t.size();) {
          size_t end = std::min(t.size(), at + part);
          if (end < t.size()) {
            const size_t nl = t.rfind('\n', end);
            if (nl != std::string::npos && nl > at + part / 2) end = nl + 1;
            while (end < t.size() && end > at && ((unsigned char)t[end] & 0xC0) == 0x80) --end;  // (not inside a character)
          }
          pts.push_back({f, l, t.substr(at, end - at)});
          at = end;
        }
      }
      // the section a part goes on with (its heading in a part before), or ""
      auto continues = [&](size_t f) {
        if (section_head(f)) return std::string();
        for (size_t k = f; k-- > 0;)
          if (section_head(k)) return ix->doc.paras[k].text.substr(0, 120);
        return std::string();
      };
      // a part's pages: a PDF's own, else an OCR .md's "Page N / M" headings
      auto pages_of = [&](size_t f, size_t l) {
        int a0 = 0, b0 = 0;
        for (size_t k = f; k < l; ++k) {
          int p = 0, m = 0;
          if (ix->doc.pages > 1) p = ix->doc.paras[k].page;
          else if (ix->doc.paras[k].level && sscanf(ix->doc.paras[k].text.c_str(), "Page %d / %d", &p, &m) < 1) p = 0;
          if (p <= 0) continue;
          if (!a0) a0 = p;
          b0 = p;
        }
        return !a0 ? std::string() : a0 == b0 ? std::to_string(a0) : std::to_string(a0) + "-" + std::to_string(b0);
      };
      // the notes file "<file name> (notes).md" (the extension kept: report.pdf and report.docx have their own): its
      // first line names the file's id, the part size and the focus it was made with. Finished notes of the same file
      // are used whatever the focus or part size (a later turn's focus in other words, a context that changed: the
      // notes were thrown away and the document read again, minutes); unfinished ones go on with the same part size
      const std::string chat = a["chat_id"].str();
      std::filesystem::path notes_path;
      // (a project's file "docs/plan.md": "plan.md (project 1a2b3c4d) (notes).md", in the chat's folder itself and not
      //  the same file as a chat upload of that name)
      std::string notes_name = i.name;
      if (i.id.rfind("project:", 0) == 0) {
        uint32_t h = 2166136261u;
        for (unsigned char c : i.id) h = (h ^ c) * 16777619u;
        char b[12];
        snprintf(b, sizeof b, "%08x", h);
        notes_name = std::filesystem::path(i.name).filename().string() + " (project " + b + ")";
      }
      if (files::valid_id(chat)) notes_path = std::filesystem::path(st.chat_dir_make(chat)) / (notes_name + " (notes).md");
      const std::string same_file = "<!-- digest of " + i.id + ",", same_parts = same_file + " parts of " + std::to_string(part) + " characters,";
      std::string mark = same_parts + " focus: ";
      for (char c : focus) mark += c == '\n' ? ' ' : c == '-' ? '_' : c;  // (no "-->" inside the comment)
      mark += " -->";
      std::string notes, old;
      size_t done = 0, done_parts = pts.size();
      bool reused = false;
      if (!notes_path.empty() && files::Store::read(notes_path.string(), old) && old.rfind(same_file, 0) == 0) {
        const size_t eol = old.find("-->");
        const bool finished = old.find("\n<!-- done -->") != std::string::npos;
        if (finished || old.rfind(same_parts, 0) == 0) {
          notes = eol == std::string::npos ? std::string() : old.substr(eol + 3);
          for (size_t p = notes.find("\n## Part "); p != std::string::npos; p = notes.find("\n## Part ", p + 1)) ++done;
          if (finished) { done_parts = done; done = pts.size(); reused = true; }
        }
      }
      if (!notes_path.empty() && notes.empty()) std::ofstream(notes_path, std::ios::binary | std::ios::trunc) << mark;
      const int max_tok = std::clamp(ctx > 0 ? ctx / 8 : 1024, 300, 1500);
      const auto t0 = std::chrono::steady_clock::now();
      for (size_t p = done; p < pts.size(); ++p) {
        if (rs.client_gone()) return;  // (the web UI's Stop: the notes so far stay, the next call goes on)
        const std::string pg = pages_of(pts[p].first, pts[p].last);
        const std::string head = "Part " + std::to_string(p + 1) + " of " + std::to_string(pts.size()) + (pg.empty() ? "" : " (pages " + pg + ")");
        // (a piece of one long paragraph after the first: it goes on with that paragraph)
        const std::string sec = p > 0 && pts[p].first == pts[p - 1].first ? ix->doc.paras[pts[p].first].text.substr(0, 120)
                                                                           : continues(pts[p].first);
        const std::string prompt =
            "Here is " + head + " of the document \"" + i.name + "\".\n" +
            (sec.empty() || code ? "" : "It begins in the middle of the section \"" + sec + "\", which started in the part "
                                        "before: note its points here as that section's.\n") +
            (sec.empty() || !code ? "" : "It begins inside the definition that starts with \"" + sec.substr(0, sec.find('\n')) +
                                         "\", in the part before.\n") +
            (focus.empty() ? "" : "What the user wants from the whole document: " + focus + "\n") +
            (code ? "Write notes of this part of the source file, to answer from later without the file: each function, "
                    "class or type with its signature and what it does, the important constants and settings, what it "
                    "calls in other parts, and whatever bears on what the user wants. Bullet points, at most about 350 "
                    "words. Only what this part holds: no introduction, no conclusion."
                  : "Write notes of this part, to answer from later without the document: its main points, facts, numbers, "
                    "names, dates, decisions, requirements, and whatever bears on what the user wants. Bullet points, in "
                    "the document's own language, at most about 350 words. Only what this part says: no introduction, no "
                    "conclusion.") +
            "\n\n---\n\n" + pts[p].text;
        std::string got;
        try {
          got = vis.ask(prompt, max_tok);
        } catch (const std::exception& e) {
          return fail(502, "part " + std::to_string(p + 1) + ": " + e.what() + " (the notes so far are kept: call again to go on)");
        }
        const std::string piece = "\n\n## " + head + "\n\n" + got;
        notes += piece;
        if (!notes_path.empty()) std::ofstream(notes_path, std::ios::binary | std::ios::app) << piece;
        zlog::eprintf("digest_file: %s, %s read (%.0f s so far)\n", i.name.c_str(), head.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
      if (!notes_path.empty() && notes.find("\n<!-- done -->") == std::string::npos) {
        std::ofstream(notes_path, std::ios::binary | std::ios::app) << "\n\n<!-- done -->\n";
        st.refresh();  // (in the Files panel)
      }
      // notes longer than one read: combined again, a few parts' notes at a time
      std::string out = notes;
      const size_t dn = out.find("\n<!-- done -->");
      if (dn != std::string::npos) out.erase(dn);
      // (until they fit one read; a level that does not shrink them ends it, and what is still too long is cut below)
      for (int level = 0; ctx > 0 && out.size() > budget && level < 6; ++level) {
        std::string next;
        for (size_t at = 0; at < out.size();) {
          size_t end = std::min(out.size(), at + part);
          if (end < out.size()) {
            const size_t cut = out.rfind("\n## ", end);
            if (cut != std::string::npos && cut > at) end = cut;
            while (end > at + 1 && ((unsigned char)out[end] & 0xC0) == 0x80) --end;  // (not inside a character)
          }
          if (rs.client_gone()) return;
          try {
            next += "\n\n" + vis.ask("Combine these notes of the document \"" + i.name + "\" into shorter notes, keeping " +
                                         "every fact, number and name that matters" +
                                         (focus.empty() ? "" : " (what the user wants: " + focus + ")") +
                                         ", and the part / page labels. Bullet points, in the notes' own language.\n\n---\n\n" +
                                         out.substr(at, end - at),
                                     max_tok);
          } catch (const std::exception& e) {
            return fail(502, std::string("combining the notes: ") + e.what());
          }
          at = end;
        }
        if (next.size() >= out.size() * 9 / 10) break;  // (not shorter: the notes before it kept, cut below)
        out = next;
      }
      std::string cut_note;
      if (ctx > 0 && out.size() > budget) {  // (never past one read: the result goes into the model's context)
        size_t end = budget;
        while (end > 0 && ((unsigned char)out[end] & 0xC0) == 0x80) --end;
        out.resize(end);
        cut_note = "; the notes are cut here to fit your context" + std::string(notes_path.empty() ? "" : ", all of them are in " +
                   notes_path.filename().string() + " (read_file it by offset)");
      }
      r.set("parts", (double)(reused ? done_parts : pts.size()));
      if (!notes_path.empty()) r.set("notes_file", notes_path.filename().string());
      r.set("content", out);
      r.set("note", "notes of every part of the document, made from its whole text" +
                        std::string(reused ? " (from the notes file made before: no part read again)"
                                    : done ? " (" + std::to_string(done) + " part(s) from the notes file made before)" : "") +
                        ": answer from them, say which part / pages a point comes from; for exact wording read_file those pages" +
                        cut_note);
      return reply(200, r);
    }
    if (rq.path == "/v1/tools/read_file") {
      files::Info i;
      Json r = Json::object();
      std::shared_ptr<const rdoc::Index> ix;
      if (files::is_virtual(a["file_id"].str())) {  // (another chat, or a transcript: read as a document)
        files::VirtualDoc v;
        if (!files::virtual_doc(a["file_id"].str(), v)) return fail(404, "no " + a["file_id"].str() + " (search_chats without a query lists the chats)");
        i.id = a["file_id"].str();
        i.name = v.title;
        i.kind = "document";
        r.set("file_id", i.id);
        r.set("title", v.title);
        r.set("kind", v.kind);
        r.set(v.kind == "chat" ? "exchanges" : "minutes", (double)v.parts);
        r.set("updated", when(v.at));
        ix = v.ix;
      } else {
        if (!find_file(i)) return fail(404, !proj_err.empty() ? proj_err : "no file " + a["file_id"].str() + " (list_files shows the saved files)");
        r.set("file_id", i.id);
        r.set("filename", i.name);
        r.set("kind", i.kind);
        r.set("bytes", (double)i.bytes);
        r.set("uploaded", when(i.created));
        std::string src;
        if (st.source(i.id, src)) {  // (the model's own document: its source)
          Json sj;
          std::string note = "a file written with create_file: 'source' holds its arguments; to correct it, change them and call create_file again";
          if (Json::try_parse(src, sj)) {
            // (its content within the model's read budget too, in parts by offset: a 60 KB page read whole overflowed
            //  a small context)
            const std::string body = sj["content"].str();
            // (as_clamped: a cast of 1e308 to size_t was undefined, BUGS.md 4a)
            const size_t mx = files::read_budget(ctx), off = (size_t)a["offset"].as_clamped(0, 0, (int64_t)body.size());
            if (off > 0 || body.size() > mx) {
              size_t b = off, end = std::min(body.size(), off + mx);
              while (b < body.size() && ((unsigned char)body[b] & 0xC0) == 0x80) ++b;
              while (end < body.size() && end > b && ((unsigned char)body[end] & 0xC0) == 0x80) --end;
              sj.set("content", body.substr(b, end - b));
              r.set("offset", (double)b);
              r.set("total_chars", (double)body.size());
              if (end < body.size()) {
                r.set("next_offset", (double)end);
                note += "; content is a part (as much as the model's context takes at once): read_file with offset: " +
                        std::to_string(end) + " for the rest; a corrected version needs the whole content again";
              }
            }
            r.set("source", sj);
          }
          r.set("note", note);
          return reply(200, r);
        }
        if (i.kind != "document") {
          r.set("note", i.kind == "image" || i.kind == "audio"
                            ? "an " + i.kind + ": its content is not text; the user can attach it to a message (@) for a model that " +
                                  (i.kind == "image" ? "sees images" : "hears audio")
                            : "a " + i.kind + " file: its content is not text");
          return reply(200, r);
        }
        ix = files::doc_index(i);
        if (!ix) return fail(500, "cannot read the file");
      }
      // the pages this call asks for ("12" or "12-15"; none: from 1, to 0)
      const bool by_pages = !a["pages"].str().empty() || a["pages"].is_number();
      const std::string pg = a["pages"].is_number() ? std::to_string(a["pages"].as_clamped(1, 1, 1 << 24)) : a["pages"].str();
      // (page numbers of at most 7 digits: the cast and atoi of a huge one were undefined, BUGS.md 4a)
      auto page_num = [](const char* q) {
        long v = 0;
        int nd = 0;
        while (*q == ' ') ++q;
        for (; isdigit((unsigned char)*q); ++q)
          if (++nd <= 7) v = v * 10 + (*q - '0');
        return nd <= 7 ? (int)v : 1 << 24;
      };
      const int from = std::max(1, page_num(pg.c_str())),
                to = !by_pages ? 0 : pg.find('-') != std::string::npos ? std::max(from, page_num(pg.c_str() + pg.find('-') + 1)) : from;
      // scanned pages among them, not transcribed yet: the loaded vision model reads them now (kScanPagesPerCall)
      // (a project's file: not here, as their text is kept by file id in a chat's folder; copied into the chat, it is)
      const std::string seer = vis.model && i.id.rfind("project:", 0) != 0 ? vis.model() : "";
      std::vector<int> want;
      for (int p : ix->doc.unread)
        if (p >= from && p <= to) want.push_back(p);
      if (!want.empty() && !seer.empty() && vis.see) {
        const std::vector<int> now(want.begin(), want.begin() + std::min<size_t>(want.size(), files::kScanPagesPerCall));
        r.set("transcribed", files::transcribe_pages(i, now, seer, vis.see));
        r.set("transcribed_by", seer);
        ix = files::doc_index(i);
        if (!ix) return fail(500, "cannot read the file");
      }
      std::string scan_note;  // (scanned pages left: how to read them)
      if (!ix->doc.unread.empty()) {
        const std::string left = rdoc::pdfimg::ranges(ix->doc.unread);
        scan_note = !seer.empty() ? "scanned pages not transcribed yet: " + left + ". read_file with pages (e.g. \"" +
                                        std::to_string(ix->doc.unread[0]) + "\"; up to " + std::to_string(files::kScanPagesPerCall) +
                                        " pages a call, 10-60 s a page) has the vision model transcribe them; afterwards search_file finds their text too"
                                  : "scanned pages " + left + " are images: the loaded model has no image encoder, so they cannot be read. "
                                    "Tell the user to load a vision model (Qwen 3.5 4B / 9B, Gemma 4 E4B / 12B / 26B with its mmproj) and ask again";
      }
      if (ix->doc.paras.empty()) {
        r.set("note", (ix->doc.note.empty() ? "no text in the file" : ix->doc.note) + (scan_note.empty() ? "" : "; " + scan_note));
        if (ix->doc.pages > 0) r.set("pages", (double)ix->doc.pages);
        return reply(200, r);
      }
      if (!ix->doc.note.empty() || !scan_note.empty())
        r.set("warning", ix->doc.note + (ix->doc.note.empty() || scan_note.empty() ? "" : "; ") + scan_note);
      if (ix->doc.pages > 0) r.set("pages", (double)ix->doc.pages);
      std::string data;  // (what this call reads: a section, pages, or the whole text)
      const size_t n = ix->doc.paras.size();
      // a text without pages of its own whose headings are "Page N / M" (an OCR model's .md, tools/server_ocr.h): its
      // pages are those sections, from page `from`'s heading up to the heading after page `to` (2026-10-31: pages
      // "26-30" of such a file said "no text", and Qwen 3.5 4B read it a page-section at a time, 8 tool rounds)
      size_t md_first = n, md_last = n;
      // the rest of the text from the call's offset in this read, when the model's context takes it in one and no
      // max_chars was asked (a model reading "offset: 0, 11934, 23882..." in 12,000-character parts ran out of its
      // tool rounds at 80K of 98K, 2026-10-31)
      auto fits = [&](const std::string& d) {
        const size_t off = std::min(d.size(), (size_t)a["offset"].as_clamped(0, 0, (int64_t)d.size()));
        return ctx > 0 && !a["max_chars"].is_number() && d.size() - off <= files::read_budget(ctx);
      };
      bool whole = false;
      if (by_pages && ix->doc.pages <= 1) {
        for (const rdoc::Section& s : ix->outline) {
          int k = 0, m = 0;
          if (sscanf(s.title.c_str(), "Page %d / %d", &k, &m) != 2) continue;
          if (k >= from && k <= to) md_first = std::min(md_first, s.first);
          else if (k > to && md_first < n) { md_last = s.first; break; }
        }
      }
      if (a["section"].is_number()) {
        const size_t k = (size_t)a["section"].as_clamped(1, 1, 1 << 30) - 1;  // (clamped before the cast)
        if (k >= ix->outline.size()) return fail(400, "no section " + std::to_string(k + 1) + " (the outline has " + std::to_string(ix->outline.size()) + ")");
        const rdoc::Section& s = ix->outline[k];
        data = ix->text(s.first, s.span);  // (with its subsections)
        r.set("section", (double)(k + 1));
        r.set("title", s.title);
        if (ix->doc.pages > 1) r.set("pages_read", files::pages_of(s));
        Json subs = Json::array();  // (its subsections' numbers, to read one of them alone)
        for (size_t j = k + 1; j < ix->outline.size() && ix->outline[j].first < s.span && subs.size() < 40; ++j) {
          Json x = Json::object();
          x.set("section", (double)(j + 1));
          x.set("title", ix->outline[j].title);
          subs.push(x);
        }
        if (subs.size()) r.set("subsections", subs);
        size_t next = k + 1;
        while (next < ix->outline.size() && ix->outline[next].first < s.span) ++next;
        if (next < ix->outline.size()) r.set("next_section", (double)(next + 1));
      } else if (by_pages && md_first < n) {
        data = ix->text(md_first, md_last);
        r.set("pages_read", pg);
      } else if (by_pages) {
        size_t first = n, last = 0;
        for (size_t k = 0; k < n; ++k)
          if (ix->doc.paras[k].page >= from && ix->doc.paras[k].page <= to) { first = std::min(first, k); last = k + 1; }
        if (first >= last && !scan_note.empty() &&
            std::any_of(ix->doc.unread.begin(), ix->doc.unread.end(), [&](int p) { return p >= from && p <= to; })) {
          r.set("pages_read", pg);
          r.set("note", "no text read on page" + std::string(from == to ? " " : "s ") + pg + ": " + scan_note);
          return reply(200, r);
        }
        if (first >= last && r.has("transcribed")) {  // (the vision model saw no text there)
          r.set("pages_read", pg);
          r.set("note", "the vision model found no text on page" + std::string(from == to ? " " : "s ") + pg);
          return reply(200, r);
        }
        if (first >= last) return fail(400, "no text on page" + std::string(from == to ? " " : "s ") + pg + (ix->doc.pages > 0 ? " (the document has " + std::to_string(ix->doc.pages) + ")" : ""));
        data = ix->text(first, last);
        r.set("pages_read", pg);
      } else {
        data = ix->text(0, n);
        // a long document: its outline first; one the model's context takes in a single read comes whole (the OCR
        // .md of 56 pages, 25K tokens, for a 262K-token Qwen 3.5 4B: section after section ran out of tool rounds)
        whole = fits(data);
        // (the outline also for a text under kInlineChars that one read cannot hold: a small context read it in parts
        //  without knowing its sections, 4 rounds for a 20K-character document)
        if ((data.size() > files::kInlineChars || (ctx > 0 && data.size() > files::read_budget(ctx))) && !whole &&
            !a["offset"].is_number()) {
          r.set("total_tokens", (double)files::tokens_of(data.size()));
          r.set("outline", files::outline_json(*ix));
          size_t cut = std::min(data.size(), (size_t)2000);
          while (cut < data.size() && cut > 0 && ((unsigned char)data[cut] & 0xC0) == 0x80) --cut;
          r.set("beginning", data.substr(0, cut));
          // (said as a must: Gemma 4 E4B, asked about all of a 159 KB file, answered from this outline alone, 2026-10-31)
          r.set("note", "this is only the outline and the beginning, not the document's text: do not answer from it. Read "
                        "the sections the question needs (read_file with section: N, or pages: \"A-B\" for a range of pages "
                        "in one call), or find passages with search_file; for a question about all of it (a summary, its "
                        "parts, a review) call digest_file now: it has you read it part by part and returns notes of every part");
          return reply(200, r);
        }
      }
      // (a part at a time: max_chars, and never more than the model's context takes in one read)
      whole = whole || fits(data);  // (a section's or pages' text, or the text from an offset, likewise)
      const size_t asked = whole ? data.size() : (size_t)std::clamp(a["max_chars"].as_number(12000), 500.0, 60000.0);
      size_t off = (size_t)a["offset"].as_clamped(0, 0, (int64_t)data.size());
      while (off < data.size() && ((unsigned char)data[off] & 0xC0) == 0x80) ++off;  // (a character's start: BUGS.md 9)
      const size_t mx = std::min(asked, files::read_budget(ctx));
      size_t end = std::min(data.size(), off + mx);
      if (end < data.size()) {  // (cut at a paragraph's end, else a line's or a space, when one is in the last third)
        size_t at = data.rfind("\n\n", end);
        if (at == std::string::npos || at <= off + mx * 2 / 3) at = data.rfind(' ', end);
        if (at != std::string::npos && at > off + mx * 2 / 3) end = at;
      }
      while (end < data.size() && end > off && ((unsigned char)data[end] & 0xC0) == 0x80) --end;  // (not inside a UTF-8 character)
      r.set("content", off < data.size() ? data.substr(off, end - off) : std::string());
      r.set("offset", (double)off);
      r.set("total_chars", (double)data.size());
      if (end < data.size()) {
        r.set("next_offset", (double)end);
        r.set("note", "a part of " + std::to_string(files::tokens_of(data.size())) + " tokens" +
                          (mx < asked ? " (as much as the model's context takes at once)" : "") +
                          ": read on with offset: " + std::to_string(end) + " only when the question needs the rest (all of "
                          "it, for a summary or a review: digest_file)");
      }
      return reply(200, r);
    }
    if (rq.path == "/v1/tools/run_command") {
      if (!terminal) return fail(404, "the terminal tool is off (--no-terminal)");
      if (!rq.local_peer) return fail(403, "run_command is only for this computer (the web UI on 127.0.0.1)");
      const std::string cmd = a["command"].str();
      const bool session = a["session"].is_bool() && a["session"].as_bool(), reset = a["reset"].is_bool() && a["reset"].as_bool();
      if (cmd.empty() && !reset && !session) return fail(400, "command required");
      // (Windows: a command line holds 32767 characters, and the script goes in as base64 of UTF-16, 2.7 characters a
      //  byte. Linux runs a script file, without that limit; the same 8000 keep one command a size a model can review)
      if (cmd.size() > 8000) return fail(400, "command too long (8000 characters at most): split it, e.g. write a script with Set-Content in parts, then run it");
      const std::string chat = a["chat_id"].str();
      const std::string cwd = files::valid_id(chat) ? st.chat_dir_make(chat) : app_dir();
      // (a project's chat: its terminal starts in the project's folder; the session's state stays in the chat's)
      std::string start = cwd;
      {
        proj::Project p;
        std::error_code e2;
        if (memo::valid_project(a["project_id"].str()) && proj::find(a["project_id"].str(), p) &&
            std::filesystem::is_directory(std::filesystem::path(p.folder), e2))
          start = p.folder;
      }
      // session: the chat's terminal (its folder and environment carried from one command to the next, kept in the
      // chat's folder); reset: that terminal closed, the next one starts fresh in the chat's (or project's) folder
      const std::filesystem::path state = std::filesystem::path(cwd) / ".terminal.json";
      std::error_code ec;
      if (reset) std::filesystem::remove(state, ec);
      auto where = [&](Json& r) {  // (the session's folder and venv now)
        std::string s, dir = start, venv;
        Json j;
        if (session && files::Store::read(state.string(), s) && Json::try_parse(s.rfind("\xEF\xBB\xBF", 0) == 0 ? s.substr(3) : s, j)) {
          if (!j["cwd"].str().empty()) dir = j["cwd"].str();
          venv = j["env"]["VIRTUAL_ENV"].str();
        }
        r.set("cwd", dir);
        if (!venv.empty()) r.set("venv", venv);
      };
      if (cmd.empty()) {  // (only the reset, or where the session is)
        Json r = Json::object();
        where(r);
        return reply(200, r);
      }
      const int tmo = (int)std::clamp(a["timeout_s"].as_number(120), 1.0, 600.0);
      std::string path = term::exe_dir();
#ifdef _WIN32
      const char* const list_sep = ";";
      const char* const filter_hint = "(Select-Object -First N, Select-String, a file)";
#else
      const char* const list_sep = ":";  // (PATH's separator on Linux)
      const char* const filter_hint = "(head -n N, grep, a file)";
#endif
      if (std::filesystem::is_directory(std::filesystem::path(app_dir()) / "bin", ec)) path = (std::filesystem::path(app_dir()) / "bin").string() + list_sep + path;
      // (the page closed the request, Stop or Esc: the command ends with it, not at its timeout)
      term::Result t = term::run(cmd, start, path, tmo, session ? state.string() : "", [&] { return rs.client_gone(); });
      if (t.cancelled) return;  // (nobody reads the reply)
      // the output as the model can take it: within the same share of its context as a read_file part (64 KB is
      // ~16K tokens, more than a small model's whole context): the start and the end of a longer one
      const size_t budget = files::read_budget(ctx);
      if (t.output.size() > budget) {
        size_t head = budget * 3 / 4, tail = t.output.size() - (budget - head);
        while (head > 0 && ((unsigned char)t.output[head] & 0xC0) == 0x80) --head;
        while (tail < t.output.size() && ((unsigned char)t.output[tail] & 0xC0) == 0x80) ++tail;
        t.output = t.output.substr(0, head) + "\n[... " + std::to_string(tail - head) + " characters left out: a smaller output " +
                   filter_hint + " shows the part you need ...]\n" + t.output.substr(tail);
        t.truncated = true;
      }
      Json r = Json::object();
      r.set("exit_code", (double)t.exit_code);
      r.set("output", t.output);
      r.set("timed_out", t.timed_out);
      r.set("truncated", t.truncated);
      r.set("seconds", std::round(t.seconds * 10) / 10);
      where(r);
      return reply(200, r);
    }
    return fail(404, "no such tool");
  } catch (const std::exception& e) {
    return fail(500, e.what());
  }
}

// /v1/files (POST multipart upload, GET list with the chats using each file), /v1/files/<id> (DELETE),
// /v1/files/delete_unused (POST), /files/<id> (GET the bytes; ?download=1 as an attachment), /v1/chats (GET list),
// /v1/chats/<id> (GET, PUT the web UI's JSON, DELETE: its folder with its files, unless ?keep_files=1)
static void files_api(const HttpRequest& rq, HttpResponse& rs) {
  auto& st = files::store();
  auto fail = [&](int code, const std::string& m) {
    Json e = Json::object(), r = Json::object();
    e.set("message", m);
    r.set("error", e);
    rs.send(code, "application/json", r.dump());
  };
  auto q = [&](const std::string& k) { return ("&" + rq.query + "&").find("&" + k + "=1&") != std::string::npos; };
  try {
    if (rq.path == "/v1/files" && rq.method == "POST") {
      const auto form = apic::parse_multipart(rq.header("content-type"), rq.body);
      const auto f = form.find("file");
      if (f == form.end() || f->second.data.empty()) return fail(400, "file required (multipart/form-data, field \"file\")");
      std::string chat;  // (the chat's folder: a form field chat_id, or ?chat_id=)
      if (auto c = form.find("chat_id"); c != form.end()) chat = c->second.data;
      else if (const size_t k = ("&" + rq.query).find("&chat_id="); k != std::string::npos) chat = rq.query.substr(k + 8, rq.query.find('&', k + 8) - k - 8);
      if (!chat.empty() && !files::valid_id(chat)) return fail(400, "bad chat_id");
      return rs.send(200, "application/json", files::info_json(st.put(f->second.filename, f->second.data, chat)).dump());
    }
    if (rq.path == "/v1/files" && rq.method == "GET") {
      const auto uses = st.uses();
      std::map<std::string, std::string> titles;  // (chat id -> title, for "used by")
      for (const auto& [cid, text] : st.chats_all()) {
        try { titles[cid] = Json::parse(text)["title"].str(); } catch (const std::exception&) { titles[cid] = cid; }
      }
      Json d = Json::array();
      double total = 0;
      for (const files::Info& i : st.list()) {
        Json j = files::info_json(i), by = Json::array();
        std::vector<std::string> cids = st.holders(i.id);
        if (auto it = uses.find(i.id); it != uses.end())
          for (const std::string& cid : it->second)
            if (std::find(cids.begin(), cids.end(), cid) == cids.end()) cids.push_back(cid);
        for (const std::string& cid : cids) {
          if (!titles.count(cid)) continue;  // (a folder whose chat was deleted: its files are unused)
            Json c = Json::object();
            c.set("id", cid);
            c.set("title", titles.count(cid) ? titles[cid] : cid);
            by.push(c);
          }
        j.set("used_by", by);
        d.push(j);
        total += (double)i.bytes;
      }
      Json r = Json::object();
      r.set("object", "list");
      r.set("data", d);
      r.set("total_bytes", total);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/v1/files/delete_unused" && rq.method == "POST") {
      // (a body {"ids": [...]}: only those files, e.g. the ones of a deleted message; no body: every file)
      std::set<std::string> only;
      Json b;
      const bool limited = !rq.body.empty() && Json::try_parse(rq.body, b) && b["ids"].is_array();  // ([] : none)
      if (!rq.body.empty() && !limited) return fail(400, "expected no body (every unused file) or {\"ids\": [...]}");
      if (limited)
        for (const Json& x : b["ids"].items()) only.insert(x.str());
      // (a file out of each folder whose chat does not refer to it any more; gone when no chat holds it)
      const auto uses = st.uses();
      int n = 0;
      double freed = 0;
      for (const files::Info& i : st.list()) {
        if (limited && !only.count(i.id)) continue;
        const auto u = uses.find(i.id);
        auto used_by = [&](const std::string& c) { return u != uses.end() && std::find(u->second.begin(), u->second.end(), c) != u->second.end(); };
        for (const std::string& c : st.holders(i.id))
          if (!used_by(c) && !st.draft(c)) st.remove_from(i.id, c);  // (a chat not saved yet: its files stay)
        if (u == uses.end()) st.remove_from(i.id, "");  // (the old uploads folder's copy)
        files::Info left;
        if (!st.get(i.id, left)) { ++n; freed += (double)i.bytes; }
      }
      Json r = Json::object();
      r.set("deleted", n);
      r.set("freed_bytes", freed);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path.rfind("/v1/files/", 0) == 0 && rq.method == "DELETE") {
      const std::string id = rq.path.substr(10);
      if (!files::valid_id(id, 16) || !st.remove(id)) return fail(404, "no file " + id);
      return rs.send(200, "application/json", "{\"deleted\":true}");
    }
    if (rq.path.rfind("/files/", 0) == 0 && rq.method == "GET") {
      files::Info i;
      const std::string id = rq.path.substr(7);
      if (!files::valid_id(id, 16) || !st.get(id, i)) return fail(404, "no file " + id);
      // (content-addressed: the same id is always the same bytes)
      // (private: a shared cache in front of a tunnel must not keep what a password protects)
      rs.extra = "Cache-Control: private, max-age=31536000, immutable\r\nX-Content-Type-Options: nosniff\r\n";
      if (q("download")) rs.extra += "Content-Disposition: attachment; filename=\"" + i.name + "\"\r\n";
      // (a page or drawing the model wrote, opened at its own address: its scripts must not run as this server's page,
      //  which can drive the tools and the remote access settings. mime_of gives these text/plain with nosniff, so they
      //  do not run; by the file's extension, in case that ever changes, the sandbox as a second layer. The web UI shows
      //  them in a sandboxed frame, see openPage)
      {
        std::string ext = std::filesystem::path(i.name).extension().string();
        for (char& ch : ext) ch = (char)tolower((unsigned char)ch);
        if (ext == ".html" || ext == ".htm" || ext == ".xhtml" || ext == ".svg" || ext == ".xml")
          rs.extra += "Content-Security-Policy: sandbox allow-scripts allow-forms allow-popups allow-modals allow-downloads\r\n";
      }
      // (read from disk 1 MB at a time, with Range: before, the whole file in memory, and 404 above 2 GB)
      return rs.send_file(std::filesystem::path(i.path), files::mime_of(i.name), rq.header("range"));
    }
    if (rq.path == "/v1/chats" && rq.method == "DELETE") {  // every chat; ?files=1: every uploaded file too
      int nc = 0, nf = 0;
      double freed = 0;
      const auto before = st.list();
      for (const auto& [cid, text] : st.chats_all())
        if (st.chat_remove(cid, !q("files"))) ++nc;
      if (q("files"))
        for (const files::Info& i : st.list()) st.remove(i.id);
      for (const files::Info& i : before) {
        files::Info left;
        if (!st.get(i.id, left)) { ++nf; freed += (double)i.bytes; }
      }
      Json r = Json::object();
      r.set("chats_deleted", nc);
      r.set("files_deleted", nf);
      r.set("freed_bytes", freed);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/v1/chats" && rq.method == "GET") {
      Json d = Json::array();
      for (const auto& [cid, text] : st.chats_all()) {
        Json c;
        try { c = Json::parse(text); } catch (const std::exception&) { continue; }  // (an unreadable file: left out)
        Json j = Json::object();
        j.set("id", cid);
        j.set("title", c["title"].str("New chat"));
        j.set("at", c["at"].as_number(0));
        j.set("model", c["model"].str(""));
        j.set("snippet", c["snippet"].str(""));
        if (!c["project"].str().empty()) j.set("project", c["project"].str());  // (tools/server_projects.h)
        d.push(j);
      }
      Json r = Json::object();
      r.set("data", d);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path.rfind("/v1/chats/", 0) == 0) {
      const std::string id = rq.path.substr(10);
      if (!files::valid_id(id)) return fail(400, "bad chat id");
      if (rq.method == "GET") {
        std::string t;
        if (!st.chat_get(id, t)) return fail(404, "no chat " + id);
        return rs.send(200, "application/json", t);
      }
      if (rq.method == "PUT" || rq.method == "POST") {
        Json c = Json::parse(rq.body);  // (checked: only JSON objects are kept)
        if (!c.is_object()) return fail(400, "a chat is a JSON object");
        st.chat_put(id, rq.body);
        return rs.send(200, "application/json", "{\"saved\":true}");
      }
      if (rq.method == "DELETE") {
        std::string t;
        if (!st.chat_get(id, t)) return fail(404, "no chat " + id);
        const auto before = st.list();
        if (!st.chat_remove(id, q("keep_files"))) return fail(404, "no chat " + id);  // (its folder, with its files)
        int n = 0;
        if (!q("keep_files")) {
          const auto uses = st.uses();
          for (const std::string& fid : files::Store::refs(t))  // (old-layout files of it that no other chat uses)
            if (!uses.count(fid) && st.holders(fid).empty()) st.remove(fid);
          for (const files::Info& i : before) {
            files::Info left;
            if (!st.get(i.id, left)) ++n;
          }
        }
        Json r = Json::object();
        r.set("deleted", true);
        r.set("files_deleted", n);
        return rs.send(200, "application/json", r.dump());
      }
    }
    return fail(405, "method not allowed");
  } catch (const std::exception& e) {
    return fail(500, e.what());
  }
}
