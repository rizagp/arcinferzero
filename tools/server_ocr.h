// OCR mode: a loaded OCR model (its name or file says "OCR": Xiaomi-OCR-0, a Qwen3.5 0.8B with an image encoder) is
// asked the way it was trained, whatever the client sends. A part of class Server (tools/server.cpp), included inside
// the class body: not a header of its own.
//
// Such a model knows a few English task prompts only. Asked in other words, behind a chat client's system prompt and
// tools, or with thinking on, it answered in Chinese or wrote a title only (2026-10-31: a letter in Indonesian, the
// README's "Extract the text in the image." word for word, "Tolong baca dan tuliskan semua teks di gambar ini." its
// title). Its text keeps the image's language either way. So a request with an image becomes: the last user turn
// alone, its image(s) and the task prompt its words ask for (the user's own text when it already is one), no system
// prompt, no tools, no thinking, greedy (the README's do_sample=False), up to 4,096 tokens.

 private:
  bool ocr_ = false;
  static constexpr uint64_t kOcrMaxImageBytes = 64ull << 20;  // (a project's image read for OCR; it is copied ~5 times)
static constexpr int kOcrMaxPages = 50;  // (a PDF's pages read per request; more: a note at the end)
  struct OcrTally {  // (tokens and timings of the pages read, added up)
    int64_t ptok = 0, ctok = 0;
    double pn = 0, cn = 0, pms = 0, gn = 0, gms = 0;
  };
  // Where the .md files go: dir; with a copy in the chat's folder (chat) when dir is a folder of the user's; in_chat:
  // dir is the chat's folder (its Files panel follows the file)
  // proj_id: dir is in that project's folder, proj_rel its path there ("ocr"; "" the folder itself)
  struct OcrDirs {
    std::filesystem::path dir, chat;
    bool in_chat = false;
    std::string proj_id, proj_rel;
  };
  // one .md the text goes into, and its copy in the chat's files (or none); ref: its "project:<id>/<path>" in a
  // project's folder
  struct OcrMd {
    std::filesystem::path md, md_chat;
    bool in_chat = false;
    std::string ref;
  };
  // A PDF in the request's last user turn (a web UI file part): every page rendered by Windows (tools/pdf_render.h)
  // and read as an image with the turn's words, page after page, each under "## Page N / M" (streamed as each page
  // is done). False when the conversation has no PDF to read. Before, a PDF went in as its text layer: no image in
  // the turn, and the follow-up rule took the conversation's last image instead, an earlier file (2026-10-31, the
  // user's LPO reports).
  //   pages: at most 50 a request, from page 1; the words may name others ("halaman 51-100", "page 7"). A turn of
  //   words alone after a PDF ("lanjut") goes on with the same PDF and task from the page after the last "## Page N"
  //   the answers so far hold; an answer that leaves pages ends with "[OCR: N more page(s) ...]", which the web UI
  //   answers with "lanjut" by itself until the last page (a 450-page manual in 9 turns).
  //   "ocr_save": true or {name, folder, append} (the web UI asks each time a PDF is sent): the pages also go into a
  //   .md, each page appended once read. name: the file's (default "<the PDF's name>.md"); folder: an existing folder
  //   of this computer, for a request from this computer only (default: the chat's folder, its Files panel); a
  //   folder of the user's gets a copy in the chat's files too, under the same name, for a chat model to read later
  //   (read_file). A new file never takes an existing one's place: "name (2).md" then; append (the web UI's "lanjut"
  //   turns): the named file goes on. Every answer ends with the file it went to ("[OCR: ... added to F]" /
  //   "[OCR: saved to F]", F "PATH; in this chat's files as NAME" for a folder of the user's).
  bool ocr_pdf(const HttpRequest& rq, const Json& body, HttpResponse& rs) {
    const Json& msgs = body["messages"];
    int last = -1;
    for (int i = 0; i < (int)msgs.size(); ++i)
      if (msgs[i]["role"].str() == "user") last = i;
    if (last < 0) return false;
    auto text_of = [](const Json& m) {
      std::string w = m["content"].is_string() ? m["content"].str() : std::string();
      for (const Json& p : m["content"].items())
        if (p["type"].str() == "text" || p["type"].str() == "input_text") w += (w.empty() ? "" : "\n") + p["text"].str();
      return w;
    };
    std::string refused;  // (a project's file this request may not read: said, not skipped)
    auto pdf_of = [&](const Json& m, files::Info& out) {
      for (const Json& p : m["content"].items()) {
        if (p["type"].str() != "file" && p["type"].str() != "input_file") continue;
        files::Info i;
        const int got = ocr_file(rq, body, p, i, refused);
        if (got < 0) return true;
        if (got > 0 && files::lower(i.name).size() > 4 && files::lower(i.name).rfind(".pdf") == files::lower(i.name).size() - 4) {
          out = i;
          return true;
        }
      }
      return false;
    };
    auto has_parts = [](const Json& m) {  // (a turn that brought a file, an image or audio: about that, not an earlier PDF)
      for (const Json& p : m["content"].items())
        if (p["type"].str() != "text" && p["type"].str() != "input_text") return true;
      return false;
    };
    files::Info pdf;
    int at = -1;  // (the turn that brought the PDF)
    for (int i = last; i >= 0 && at < 0; --i) {
      if (msgs[i]["role"].str() != "user") continue;
      if (pdf_of(msgs[i], pdf)) at = i;
      else if (i == last && has_parts(msgs[i])) break;
    }
    if (at < 0) return false;
    if (!refused.empty()) {  // (only for the turn that brings the file, as in ocr_batch)
      if (at < last) return false;
      rs.send(403, "application/json", "{\"error\":{\"message\":" + Json(refused).dump() + "}}");
      return true;
    }
    const std::string words = text_of(msgs[last]), task = text_of(msgs[at]);  // (a continuation keeps the PDF's task)
    std::string err;
    std::string bytes;
    pdfr::Pdf doc;
    if (!files::Store::read(pdf.path, bytes) && !(files::store().get(pdf.id, pdf) && files::Store::read(pdf.path, bytes)))
      err = "the file is gone";  // (once more: its chat's folder renamed meanwhile)
    else doc.open(bytes, err);
    const int pages = err.empty() ? doc.pages() : -1;
    if (pages < 0) {
      rs.send(400, "application/json", "{\"error\":{\"message\":" + Json("cannot read the PDF: " + err).dump() + "}}");
      return true;
    }
    int first = 1, upto = 0;
    ocr_pages_asked(words, first, upto);
    if (upto == 0 && at < last) {  // (a continuation: after the last page the answers so far read)
      int done = 0;
      for (int i = at + 1; i < (int)msgs.size(); ++i)
        if (msgs[i]["role"].str() == "assistant") done = std::max(done, ocr_last_page(text_of(msgs[i])));
      first = done + 1;
    }
    first = std::max(1, first);
    const int n = std::min(pages, upto ? upto : first + kOcrMaxPages - 1);
    const bool stream = body["stream"].as_bool(false);
    const std::string id = "chatcmpl-arc" + std::to_string(++counter_);
    const long long created = (long long)time(nullptr);
    std::string all;
    // the .md next to the chat's files (started over with page 1, else appended)
    OcrDirs dirs;
    OcrMd md;
    const Json& sv = body["ocr_save"];
    if (sv.as_bool(false) || sv.is_object()) {
      std::string name = sv["name"].str();
      if (name.empty()) name = std::filesystem::path(pdf.name).stem().string() + ".md";
      if (!ocr_dirs(rq, body, std::filesystem::path(pdf.path).parent_path(), dirs, rs)) return true;
      if (!ocr_md_make(dirs, ocr_md_name(name), sv["append"].as_bool(false), "# " + pdf.name + "\n\n", md, rs)) return true;
    }
    // (begun after the file checks: an error said before them went out as an event, not as the error it is)
    if (stream) rs.sse_begin();
    auto save = [&](const std::string& piece) { return ocr_md_add(md, piece); };
    // (the file named at the end of each answer: its name in the chat's files, else its full path; with the copy's
    //  name after "; in this chat's files as")
    const std::string where = ocr_md_where(md);
    zlog::eprintf("OCR mode: %s, pages %d-%d of %d%s\n", pdf.name.c_str(), first, n, pages,
            md.md.empty() ? "" : (", saved to " + md.md.filename().string()).c_str());  // (the file's name only: no user's path in the log)
    if (first > pages) {
      const std::string note = "[OCR: every page of " + pdf.name + " is read (" + std::to_string(pages) + ")]";
      if (stream) {
        Json d = Json::object();
        d.set("content", note);
        rs.sse_event(chunk(id, created, d, Json()).dump());
      } else {
        all = note;
      }
    }
    // (the pages' timings added up: the web UI's statistics line, which said "stopped" for an answer without them)
    OcrTally tally;
    std::string failed;  // (a page the model could not read: the pages stop there, no "more pages" note to go on with)
    for (int p = first; p <= n && !rs.client_gone() && failed.empty(); ++p) {
      std::string piece = (p > first ? "\n\n" : "") + std::string("## Page ") + std::to_string(p) + " / " + std::to_string(pages) + "\n\n";
      size_t sent = 0;  // (the heading streamed before the page is read: the web UI shows the page under way)
      if (stream) {
        Json d = Json::object();
        d.set("content", piece);
        if (!rs.sse_event(chunk(id, created, d, Json()).dump())) return true;
        sent = piece.size();
      }
      const std::string png = doc.render(p, 150, err);
      if (png.empty()) {
        piece += "[the page could not be rendered: " + err + "]";
      } else {
        std::string text;
        if (ocr_image(rq, rs, "data:image/png;base64," + apic::base64_encode(png), task, tally, text)) {
          piece += text;
        } else {
          failed = "page " + std::to_string(p) + ": " + text;
          piece += "[" + failed + "]";
        }
      }
      if (!save((p > 1 && p == first ? "\n\n" : "") + piece) && failed.empty())
        failed = "page " + std::to_string(p) + ": could not write " + md.md.filename().string() + " (is the disk full?)";
      if (stream) {
        Json d = Json::object();
        d.set("content", piece.substr(sent));
        if (!rs.sse_event(chunk(id, created, d, Json()).dump())) return true;
      } else {
        all += piece;
      }
    }
    std::string note;  // (pages left: the web UI's cue for "lanjut"; and the file the text went to)
    if (!failed.empty()) {  // (the web UI goes on only after a "more page(s)" note: a model that cannot read the pages, no
                            //  vision encoder, would have answered every "lanjut" with the same error, 450 times)
      note = "\n\n[OCR: stopped at " + failed + (md.md.empty() ? "" : "; the pages before it are in " + where) + "]";
    } else if (pages > n && n >= first) {
      const int a = n + 1, b = std::min(pages, n + kOcrMaxPages);
      note = "\n\n[OCR: " + std::to_string(pages - n) + " more page(s); \"lanjut\" reads pages " + std::to_string(a) + "-" +
             std::to_string(b) + (md.md.empty() ? "" : ", added to " + where) + "]";
    } else if (!md.md.empty()) {
      note = "\n\n[OCR: saved to " + where + "]";
    }
    if (!note.empty()) {
      if (stream) {
        Json d = Json::object();
        d.set("content", note);
        rs.sse_event(chunk(id, created, d, Json()).dump());
      } else {
        all += note;
      }
    }
    ocr_finish(rs, stream, id, created, all, tally);
    return true;
  }
  // The answer's end: usage and timings (the pages' added up), the last event or the whole answer
  void ocr_finish(HttpResponse& rs, bool stream, const std::string& id, long long created, const std::string& all,
                  const OcrTally& t) {
    Json usage = Json::object();
    usage.set("prompt_tokens", t.ptok);
    usage.set("completion_tokens", t.ctok);
    usage.set("total_tokens", t.ptok + t.ctok);
    Json timings = Json::object();
    timings.set("prompt_n", t.pn);
    timings.set("cache_n", t.cn);
    timings.set("prompt_ms", t.pms);
    timings.set("prompt_per_second", t.pms > 0 ? 1000.0 * t.pn / t.pms : 0.0);
    timings.set("predicted_n", t.gn);
    timings.set("predicted_ms", t.gms);
    timings.set("predicted_per_second", t.gms > 0 ? 1000.0 * t.gn / t.gms : 0.0);
    if (stream) {
      Json fin = chunk(id, created, Json::object(), Json("stop"));
      fin.set("usage", usage);
      fin.set("timings", timings);
      rs.sse_event(fin.dump());
      rs.sse_event("[DONE]");
      rs.sse_end();
      return;
    }
    Json msg = Json::object(), c = Json::object(), choices = Json::array(), r = Json::object();
    msg.set("role", "assistant");
    msg.set("content", all);
    c.set("index", 0);
    c.set("message", msg);
    c.set("finish_reason", "stop");
    choices.push(c);
    r.set("id", id);
    r.set("object", "chat.completion");
    r.set("created", (double)created);
    r.set("model", model_id_);
    r.set("choices", choices);
    r.set("usage", usage);
    r.set("timings", timings);
    rs.send(200, "application/json", r.dump());
  }

  // ---- the pieces ocr_pdf and ocr_batch share (their types: at the top)
  // a request from this computer, from the server's own page or a client that sends no Origin (not a page another
  // local program serves: such a page may call the chat API, and the connection is local for it too, so local_peer
  // alone let it write a .md into any folder; BUGS.md 1, 2026-10-31). A remote session with full access: its page is
  // https, through the tunnel.
  // no Origin (a program, not a page) or this server's own page (http here, https through the tunnel)
  static bool ocr_from_page_ok(const HttpRequest& rq) {
    std::string origin = rq.header("origin"), own = (rq.remote ? "https://" : "http://") + rq.header("host");
    for (std::string* x : {&origin, &own})
      for (char& c : *x) c = (char)tolower((unsigned char)c);
    return origin.empty() || origin == own;
  }
  static bool ocr_own_page(const HttpRequest& rq) {
    std::string origin = rq.header("origin"), own = (rq.remote ? "https://" : "http://") + rq.header("host");
    for (std::string* x : {&origin, &own})
      for (char& c : *x) c = (char)tolower((unsigned char)c);
    return rq.local_peer && (origin.empty() || origin == own);
  }
  // A message's file part as the file to read: 1 found (out), 0 none (not a file part, or no such upload), -1 refused
  // (err). An upload of the chat's (files::store) or a project's file, "project:<id>/<path>": read where it is, no
  // copy, and only as read_file reads one (tools/server_file_tools.h project_file): the request's own project
  // (project_id, which the web UI sends for a project's chat), from this computer and the server's own page.
  int ocr_file(const HttpRequest& rq, const Json& body, const Json& part, files::Info& out, std::string& err) {
    if (part["type"].str() != "file" && part["type"].str() != "input_file") return 0;
    std::string fid = part["file"]["file_id"].str(part["file_id"].str());
    const std::string given = part["file"]["filename"].is_string() ? part["file"]["filename"].str() : "";
    std::string pid, rel;
    if (proj::split_id(fid, pid, rel)) {
      proj::Project p;
      std::filesystem::path f;
      if (!ocr_own_page(rq)) err = "a project's files are only for this computer (this server's own page)";
      else if (pid != body["project_id"].str()) err = "project:" + pid + " is not this chat's project: only the files of the chat's own project can be read";
      else if (!memo::valid_project(pid) || !proj::find(pid, p)) err = "no project " + pid;
      else if (proj::resolve_in(p, rel, f, err)) {
        std::error_code ec;
        out = files::Info();
        out.id = fid;
        out.path = f.string();
        out.name = std::filesystem::path(rel).filename().string();  // (its name; the path inside the project in id)
        out.kind = files::kind_of(rel);
        out.bytes = std::filesystem::file_size(f, ec);
        return 1;
      }
      return -1;
    }
    if (fid.rfind("/files/", 0) == 0) fid = fid.substr(7);
    if (fid.rfind("file-", 0) == 0) fid = fid.substr(5);
    if (!files::store().get(fid, out)) return 0;
    // (the name the chat gave it: one file in several chats is one entry under whichever name came first)
    if (!given.empty()) out.name = ocr_utf8_ok(given) ? given : files::safe_name(given);
    return 1;
  }
  // the folder a project's chat saves its OCR text into by default: "ocr" in the project's folder (made when needed),
  // which every chat of the project lists (list_files, scope "project"); empty when the request has no project, may
  // not write there, or the folder cannot be made
  std::filesystem::path ocr_project_dir(const HttpRequest& rq, const Json& body, proj::Project& p, bool make) {
    const std::string pid = body["project_id"].str();
    if (pid.empty() || !memo::valid_project(pid) || !proj::find(pid, p) || !ocr_own_page(rq)) return {};
    std::error_code ec;
    const std::filesystem::path root = std::filesystem::path(p.folder);
    if (!std::filesystem::is_directory(root, ec)) return {};
    const std::filesystem::path d = root / "ocr";
    if (make) std::filesystem::create_directories(d, ec);  // (only when the .md goes there, not for another folder)
    return d;
  }
  // well-formed UTF-8 (a name from the request; std::filesystem::path throws on bad bytes)
  static bool ocr_utf8_ok(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
      const unsigned char c = (unsigned char)s[i];
      const int n = c < 0x80 ? 0 : (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : -1;
      if (n < 0 || c == 0xC0 || c == 0xC1 || c > 0xF4 || i + n >= s.size() + (n ? 0 : 1)) return false;
      for (int k = 1; k <= n; ++k)
        if (((unsigned char)s[i + k] >> 6) != 2) return false;
      const unsigned char d = n ? (unsigned char)s[i + 1] : 0;  // (overlong forms, UTF-16 surrogates, past U+10FFFF)
      if ((c == 0xE0 && d < 0xA0) || (c == 0xED && d >= 0xA0) || (c == 0xF0 && d < 0x90) || (c == 0xF4 && d >= 0x90)) return false;
      i += n + 1;
    }
    return true;
  }
  // a file name for a .md: a name, never a path; no character Windows refuses, no device's name, ending in .md
  static std::string ocr_md_name(std::string name) {
    name = std::filesystem::path(name).filename().string();
    for (char& c : name)
      if (strchr("<>:\"/\\|?*", c) || (unsigned char)c < 32) c = '_';
    if (name.empty() || name[0] == '.') name = "ocr" + name;
    if (files::lower(name).size() < 3 || files::lower(name).rfind(".md") != files::lower(name).size() - 3) name += ".md";
    {  // (a device's name, "CON.md", "nul.md", "COM1.md": Windows opens the device, not a file)
      const std::string st = files::lower(std::filesystem::path(name).stem().string());
      const std::string base = st.substr(0, st.find('.'));
      if (base == "con" || base == "prn" || base == "aux" || base == "nul" ||
          ((base.rfind("com", 0) == 0 || base.rfind("lpt", 0) == 0) && base.size() == 4 && isdigit((unsigned char)base[3])))
        name = "ocr " + name;
    }
    return name;
  }
  // ocr_save's folder (default: the chat's folder, else `fallback`, the folder of the file read); false after an error
  // was sent
  // A project's chat (project_id): by default the project's "ocr" folder, no copy in the chat's files (every chat of
  // the project and any model reads it there: "project:<id>/ocr/NAME"); a folder named inside the project's folder
  // likewise.
  bool ocr_dirs(const HttpRequest& rq, const Json& body, const std::filesystem::path& fallback, OcrDirs& d, HttpResponse& rs) {
    const std::string folder = body["ocr_save"]["folder"].str(), chat = body["chat_id"].str();
    // (a page another local program serves: no .md anywhere, the chat's folder neither; a client with no Origin,
    //  the server's own page and a remote session's page may)
    if (!ocr_from_page_ok(rq)) {
      rs.send(403, "application/json", "{\"error\":{\"message\":\"ocr_save: from this server's own page or a client on this computer only\"}}");
      return false;
    }
    std::filesystem::path chat_dir;
    // (the asking chat's folder: the file's copy found may be another chat's)
    if (files::valid_id(chat)) chat_dir = std::filesystem::path(files::store().chat_dir_make(chat));
    std::error_code ec;
    proj::Project p;
    const std::filesystem::path pdir = ocr_project_dir(rq, body, p, folder.empty());
    if (!folder.empty()) {
      // (this computer, and not a page another local program serves, see ocr_own_page)
      if (!ocr_own_page(rq)) {
        rs.send(403, "application/json", "{\"error\":{\"message\":\"ocr_save.folder: from this server's own page or a client on this computer only\"}}");
        return false;
      }
      d.dir = std::filesystem::path(folder);
      if (!d.dir.is_absolute() || !std::filesystem::is_directory(d.dir, ec)) {
        rs.send(400, "application/json", "{\"error\":{\"message\":" + Json("no such folder: " + folder).dump() + "}}");
        return false;
      }
      std::string rel;
      if (std::filesystem::equivalent(d.dir, chat_dir, ec)) {  // (the chat's own folder named: as an empty folder)
        d.in_chat = true;
      } else if (!pdir.empty() && ocr_in_project(p, d.dir, rel)) {  // (in the project: its chats read it there)
        d.proj_id = p.id;
        d.proj_rel = rel;
      } else {
        // a folder of the user's: a copy in the chat's files as well, under the same name, which a chat model reads
        // later (read_file NAME; it reads no folder of this computer)
        d.chat = chat_dir;
      }
    } else if (!pdir.empty() && std::filesystem::is_directory(pdir, ec)) {
      d.dir = pdir;
      d.proj_id = p.id;
      d.proj_rel = "ocr";
    } else {
      d.dir = chat_dir.empty() ? fallback : chat_dir;
      d.in_chat = true;
    }
    return true;
  }
  // dir inside the project's folder (or the folder itself): its path there with "/" ("" for the folder itself)
  static bool ocr_in_project(const proj::Project& p, const std::filesystem::path& dir, std::string& rel) {
    std::error_code ec;
    const std::filesystem::path root = std::filesystem::weakly_canonical(std::filesystem::path(p.folder), ec),
                                d = std::filesystem::weakly_canonical(dir, ec);
    if (root.empty() || d.empty()) return false;
    std::wstring r = root.wstring(), x = d.wstring();
    for (wchar_t& c : r) c = (wchar_t)towlower(c);
    for (wchar_t& c : x) c = (wchar_t)towlower(c);
    while (!r.empty() && r.back() == L'\\') r.pop_back();
    while (!x.empty() && x.back() == L'\\') x.pop_back();
    if (x.compare(0, r.size(), r) != 0 || (x.size() > r.size() && x[r.size()] != L'\\')) return false;
    rel = x.size() > r.size() ? d.lexically_relative(root).generic_string() : "";
    return true;
  }
  // The .md `name` in d (and its copy): a new file starting with `head`, never in an existing one's place nor its
  // copy's ("name (2).md" then); append: the named file goes on as it is. False after an error was sent (ocr_md_try:
  // false with the HTTP status and why, for a stream that began already).
  bool ocr_md_make(const OcrDirs& d, const std::string& name, bool append, const std::string& head, OcrMd& out, HttpResponse& rs) {
    int code = 0;
    std::string why;
    if (ocr_md_try(d, name, append, head, out, code, why)) return true;
    rs.send(code, "application/json", "{\"error\":{\"message\":" + Json(why).dump() + "}}");
    return false;
  }
  bool ocr_md_try(const OcrDirs& d, std::string name, bool append, const std::string& head, OcrMd& out, int& code, std::string& why) {
    std::error_code ec;
    out.in_chat = d.in_chat;
    out.md = d.dir / name;
    out.md_chat = d.chat.empty() ? std::filesystem::path() : d.chat / name;
    auto ref = [&] {
      out.ref = d.proj_id.empty() ? "" : "project:" + d.proj_id + "/" + (d.proj_rel.empty() ? "" : d.proj_rel + "/") + out.md.filename().string();
    };
    ref();
    if (append) return true;
    const std::string stem = out.md.stem().string();
    auto taken = [&] { return std::filesystem::exists(out.md, ec) || (!out.md_chat.empty() && std::filesystem::exists(out.md_chat, ec)); };
    for (int k = 2; taken() && k < 1000; ++k) {
      name = stem + " (" + std::to_string(k) + ").md";
      out.md = d.dir / name;
      if (!out.md_chat.empty()) out.md_chat = d.chat / name;
    }
    ref();
    // (999 names taken: the loop ended on one that exists, which the write below would have replaced)
    if (taken()) {
      code = 409;
      why = "every name from " + stem + ".md to " + stem + " (999).md is taken: choose another name";
      return false;
    }
    for (const auto& f_path : {out.md, out.md_chat}) {
      if (f_path.empty()) continue;
      std::ofstream f(f_path, std::ios::binary | std::ios::trunc);
      f << head;
      if (!f.flush()) {  // (a read-only folder, a full disk: said now, not pages read into no file)
        code = 400;
        why = "cannot write " + name + " there";
        return false;
      }
    }
    return true;
  }
  // false: the text could not be added (a full disk, the folder removed): the caller stops, saying so
  static bool ocr_md_add(const OcrMd& m, const std::string& piece) {
    if (m.md.empty()) return true;
    bool ok;
    {
      std::ofstream f(m.md, std::ios::binary | std::ios::app);
      ok = (bool)(f << piece).flush();
    }
    if (ok && !m.md_chat.empty()) {
      std::ofstream f(m.md_chat, std::ios::binary | std::ios::app);
      ok = (bool)(f << piece).flush();
    }
    if (!ok) LOG_WRN("OCR: could not add this page to %s (file open elsewhere, or disk full?): the page's text is only in the chat\n", m.md.filename().string().c_str());
    if (m.in_chat || !m.md_chat.empty()) files::store().refresh();  // (in the Files panel with its new size)
    return ok;
  }
  // a project image's type for its data URL (the model's decoder reads the bytes; the name says what they are)
  static std::string ocr_image_mime(const std::string& name) {
    std::string e = files::lower(std::filesystem::path(name).extension().string());
    if (e == ".jpg" || e == ".jpeg") return "image/jpeg";
    if (e == ".webp") return "image/webp";
    if (e == ".gif") return "image/gif";
    if (e == ".bmp") return "image/bmp";
    return "image/png";
  }
  // the file named in an answer's note: its name in the chat's files, else its full path, with the copy's name after
  // "; in this chat's files as"
  static std::string ocr_md_where(const OcrMd& m) {
    return m.md.empty() ? "" : m.in_chat ? m.md.filename().string()
                         : m.md.string() + (!m.ref.empty() ? "; in the project as " + m.ref
                                            : m.md_chat.empty() ? "" : "; in this chat's files as " + m.md_chat.filename().string());
  }
  // One image read by the model (url: a data URL or /files/<id>) with the task's words, through chat(): its text, or
  // false and why. Its tokens and timings go into t.
  bool ocr_image(const HttpRequest& rq, HttpResponse& outer, const std::string& image_url, const std::string& task, OcrTally& t, std::string& text) {
    Json url = Json::object(), img = Json::object(), txt = Json::object(), content = Json::array(), msg = Json::object(),
         one = Json::array(), req = Json::object();
    url.set("url", image_url);
    img.set("type", "image_url");
    img.set("image_url", url);
    txt.set("type", "text");
    txt.set("text", task);
    content.push(img);
    content.push(txt);
    msg.set("role", "user");
    msg.set("content", content);
    one.push(msg);
    req.set("messages", one);
    req.set("stream", false);
    HttpRequest rq2 = rq;
    rq2.body = req.dump();
    HttpResponse r2(INVALID_SOCKET);
    int st = 0;
    std::string out;
    r2.on_send = [&](int s, const std::string& b) { st = s; out = b; };
    r2.gone_hook = [&outer] { return outer.client_gone(); };  // (Stop ends the page under way, not only the next one)
    chat(rq2, r2);  // (an image now: ocr_request gives it the task prompt; its tables come back as Markdown)
    Json j;
    Json::try_parse(out, j);
    t.ptok += j["usage"]["prompt_tokens"].as_int();
    t.ctok += j["usage"]["completion_tokens"].as_int();
    const Json& tm = j["timings"];
    t.pn += tm["prompt_n"].as_number(0);
    t.cn += tm["cache_n"].as_number(0);
    t.pms += tm["prompt_ms"].as_number(0);
    t.gn += tm["predicted_n"].as_number(0);
    t.gms += tm["predicted_ms"].as_number(0);
    if (st != 200) {
      text = j["error"]["message"].str("HTTP " + std::to_string(st));
      return false;
    }
    text = j["choices"][0]["message"]["content"].str();
    return true;
  }

  // Several files at once (2026-11-08): a user turn that brought two or more images and PDFs (web UI file parts) has
  // them read in their order, each under "# File K / F: NAME" (a PDF's pages under "## Page N / M" in it), with the
  // turn's words as the task of every one. False when the conversation's latest turn with files has fewer than two.
  //   pages: at most 50 a request over all the files (an image is one), from where the answers so far stopped: a turn
  //   of words alone ("lanjut") goes on after the last file / page with text; an answer that leaves pages ends with
  //   "[OCR: N more page(s) ...]", which the web UI answers with "lanjut" by itself. Page words ("halaman 7") are not
  //   read here: they name no file.
  //   "ocr_save" {name, folder, append, per_file}: as ocr_pdf's. per_file false: one .md (name, default "<the first
  //   file's name> and N more.md"), each file under "# NAME" in it; true: one .md per file, "<its name>.md" (name not
  //   read), and append goes on with the file in progress, the one the last "added to F" note named.
  bool ocr_batch(const HttpRequest& rq, const Json& body, HttpResponse& rs) {
    const Json& msgs = body["messages"];
    int last = -1;
    for (int i = 0; i < (int)msgs.size(); ++i)
      if (msgs[i]["role"].str() == "user") last = i;
    if (last < 0) return false;
    int at = -1;  // (the latest user turn that brought anything but words)
    for (int i = last; i >= 0 && at < 0; --i) {
      if (msgs[i]["role"].str() != "user") continue;
      for (const Json& p : msgs[i]["content"].items())
        if (p["type"].str() != "text" && p["type"].str() != "input_text") at = i;
    }
    if (at < 0) return false;
    struct Item {
      files::Info info;
      bool pdf = false;
      pdfr::Pdf doc;
      int pages = 1;
      std::string err;  // (a PDF that cannot be read: one entry with why, the batch goes on)
    };
    std::vector<Item> items;
    std::string refused;  // (a project's file this request may not read)
    for (const Json& p : msgs[at]["content"].items()) {
      Item it;
      if (ocr_file(rq, body, p, it.info, refused) <= 0) continue;
      const std::string ln = files::lower(it.info.name);
      it.pdf = ln.size() > 4 && ln.rfind(".pdf") == ln.size() - 4;
      if (!it.pdf && it.info.kind != "image") continue;  // (a file of another kind: not one this model reads)
      items.push_back(std::move(it));
    }
    // (one file: ocr_pdf's or ocr_request's, but a project's image too here: it has no /files/ address for them)
    const bool proj_image = items.size() == 1 && !items[0].pdf && items[0].info.id.rfind("project:", 0) == 0;
    // (also a single file: said, not read as "a file that is no longer there"; only for the turn that brings it: a
    //  later turn about something else is not refused for an old file of the chat)
    if (!refused.empty()) {
      if (at < last) return false;
      rs.send(403, "application/json", "{\"error\":{\"message\":" + Json(refused).dump() + "}}");
      return true;
    }
    if (items.size() < 2 && !proj_image) return false;
    auto text_of = [](const Json& m) {
      std::string w = m["content"].is_string() ? m["content"].str() : std::string();
      for (const Json& p : m["content"].items())
        if (p["type"].str() == "text" || p["type"].str() == "input_text") w += (w.empty() ? "" : "\n") + p["text"].str();
      return w;
    };
    const std::string task = text_of(msgs[at]);
    const int nf = (int)items.size();
    for (Item& it : items) {
      if (it.pdf) {
        std::string bytes;
        if (!files::Store::read(it.info.path, bytes) && !(files::store().get(it.info.id, it.info) && files::Store::read(it.info.path, bytes)))
          it.err = "the file is gone";
        else if (it.doc.open(bytes, it.err) && it.doc.pages() < 1) it.err = "no pages";
        it.pages = it.err.empty() ? it.doc.pages() : 1;
      }
    }
    // where to start: file fk (0-based), page fp
    int fk = 0, fp = 1;
    if (at < last) {
      std::string s;
      for (int i = at + 1; i < (int)msgs.size(); ++i)
        if (msgs[i]["role"].str() == "assistant") s += "\n\n" + text_of(msgs[i]);
      for (size_t a; (a = s.find("[OCR:")) != std::string::npos;) {  // (the notes are not a file's text)
        const size_t b = s.find(']', a);
        s.erase(a, b == std::string::npos ? std::string::npos : b - a + 1);
      }
      // (file number, its text up to the next file's heading). A heading is a whole line "# File K / F: " with F this
      // batch's count: the model's own text ("## File menu", "# File 1 notes") is a file's text, not a heading
      // (self audit 2026-11-08: such a line cut a file's pages short, and they were read again)
      auto head_at = [&](size_t p, int& k) {
        if (p != 0 && s[p - 1] != '\n') return false;
        int f = 0, n = 0;
        if (sscanf(s.c_str() + p, "# File %d / %d: %n", &k, &f, &n) != 2 || n == 0) return false;
        return f == nf && k >= 1 && k <= nf;
      };
      auto next_head = [&](size_t from, int& k) {
        for (size_t p = s.find("# File ", from); p != std::string::npos; p = s.find("# File ", p + 1))
          if (head_at(p, k)) return p;
        return std::string::npos;
      };
      std::vector<std::pair<int, std::string>> segs;
      int hk = 0, hk2 = 0;
      for (size_t p = next_head(0, hk); p != std::string::npos;) {
        const size_t next = next_head(p + 7, hk2), eol = s.find('\n', p);
        const size_t stop = next == std::string::npos ? s.size() : next;
        segs.push_back({hk, eol == std::string::npos || eol > stop ? std::string() : s.substr(eol, stop - eol)});
        p = next;
        hk = hk2;
      }
      for (int i = (int)segs.size() - 1; i >= 0; --i) {
        const auto& [k, body_] = segs[i];
        if (k < 1 || k > nf || !ocr_has_text(body_)) continue;
        const Item& it = items[k - 1];
        const int done = it.pdf && it.err.empty() ? ocr_last_page(body_) : it.pages;
        if (done >= it.pages) fk = k, fp = 1;
        else fk = k - 1, fp = done + 1;
        break;
      }
    }
    const Json& sv = body["ocr_save"];
    const bool saving = sv.as_bool(false) || sv.is_object(), per_file = sv["per_file"].as_bool(false);
    OcrDirs dirs;
    OcrMd md;  // (the .md written now: the merged one, or the file in progress's)
    std::vector<std::string> made;  // (per file: the .md files this answer started)
    if (saving) {
      if (!ocr_dirs(rq, body, std::filesystem::path(items[0].info.path).parent_path(), dirs, rs)) return true;
      if (!per_file) {
        std::string name = sv["name"].str();
        if (name.empty())
          name = std::filesystem::path(items[0].info.name).stem().string() + " and " + std::to_string(nf - 1) + " more.md";
        if (!ocr_md_make(dirs, ocr_md_name(name), sv["append"].as_bool(false), "", md, rs)) return true;
      } else if (fk < nf) {
        // the file in progress goes on in its .md: the name the last note gave, when it is that file's ("<its name>.md"
        // or "<its name> (N).md"). An answer stopped by the user ends with no note, and the name the web UI still has
        // is an earlier file's: then a new .md (self audit 2026-11-08: the pages went into the earlier file's .md)
        const std::string want = files::lower(std::filesystem::path(items[fk].info.name).stem().string()),
                          got = files::lower(ocr_md_name(sv["name"].str()));
        bool its = got == want + ".md";
        if (!its && got.size() > want.size() + 6 && got.compare(0, want.size() + 2, want + " (") == 0 && got.rfind(").md") == got.size() - 4) {
          its = true;
          for (size_t i = want.size() + 2; i < got.size() - 4; ++i) its = its && isdigit((unsigned char)got[i]);
        }
        const bool goes_on = fp > 1 && sv["append"].as_bool(false) && its;
        // (begun here, before the stream: a folder that cannot be written is an HTTP error, not one inside the stream)
        if (!ocr_md_make(dirs, goes_on ? ocr_md_name(sv["name"].str()) : ocr_md_name(std::filesystem::path(items[fk].info.name).stem().string() + ".md"),
                         goes_on, goes_on ? "" : "# " + items[fk].info.name + "\n\n", md, rs))
          return true;
        if (!goes_on) made.push_back(md.md.filename().string());
      }
    }
    const bool stream = body["stream"].as_bool(false);
    const std::string id = "chatcmpl-arc" + std::to_string(++counter_);
    const long long created = (long long)time(nullptr);
    if (stream) rs.sse_begin();
    std::string all;
    bool gone = false;
    auto emit = [&](const std::string& t) {
      if (t.empty() || gone) return;
      if (!stream) { all += t; return; }
      Json d = Json::object();
      d.set("content", t);
      if (!rs.sse_event(chunk(id, created, d, Json()).dump())) gone = true;
    };
    zlog::eprintf("OCR mode: %d files, from file %d page %d%s\n", nf, fk + 1, fp,
            !saving ? "" : per_file ? ", one .md each" : (", saved to " + md.md.filename().string()).c_str());
    OcrTally tally;
    std::string failed;
    int read = 0, k = fk, p = fp;  // (k, p: the next page to read)
    bool any = false;              // (something emitted: the pieces after it start with a blank line)
    if (fk >= nf) emit("[OCR: every file is read (" + std::to_string(nf) + ")]");
    for (; k < nf && failed.empty() && !gone && !rs.client_gone(); ++k, p = 1) {
      Item& it = items[k];
      if (read >= kOcrMaxPages) break;
      emit(std::string(any ? "\n\n" : "") + "# File " + std::to_string(k + 1) + " / " + std::to_string(nf) + ": " + it.info.name + "\n\n");
      any = true;
      if (saving && per_file && k > fk) {  // (each next file's .md; the first one's was made before the stream)
        int code = 0;
        std::string why;
        if (!ocr_md_try(dirs, ocr_md_name(std::filesystem::path(it.info.name).stem().string() + ".md"), false,
                        "# " + it.info.name + "\n\n", md, code, why)) {
          md = OcrMd();  // (the note names no file that was not made)
          failed = "file " + std::to_string(k + 1) + ": " + why;
          emit("[" + failed + "]");
          break;
        }
        made.push_back(md.md.filename().string());
      } else if (saving && !per_file && p == 1) {
        if (!ocr_md_add(md, "# " + it.info.name + "\n\n")) {
          failed = "file " + std::to_string(k + 1) + ": could not write " + md.md.filename().string() + " (is the disk full?)";
          emit("[" + failed + "]");
          break;
        }
      }
      if (!it.err.empty()) {
        const std::string t = "[cannot read the PDF: " + it.err + "]";
        emit(t);
        if (!ocr_md_add(md, t + "\n\n")) {
          failed = "file " + std::to_string(k + 1) + ": could not write " + md.md.filename().string() + " (is the disk full?)";
          emit("[" + failed + "]");
          break;
        }
        ++read;
        continue;
      }
      for (; p <= it.pages && read < kOcrMaxPages && failed.empty() && !gone && !rs.client_gone(); ++p, ++read) {
        std::string piece;
        if (it.pdf) {
          piece = std::string(p > (k == fk ? fp : 1) ? "\n\n" : "") + "## Page " + std::to_string(p) + " / " + std::to_string(it.pages) + "\n\n";
          emit(piece);  // (the heading before the page is read: the web UI shows the page under way)
        }
        std::string text, err;
        bool ok = false;
        if (it.pdf) {
          const std::string png = it.doc.render(p, 150, err);
          if (png.empty()) text = "[the page could not be rendered: " + err + "]", ok = true;
          else ok = ocr_image(rq, rs, "data:image/png;base64," + apic::base64_encode(png), task, tally, text);
        } else if (it.info.id.rfind("project:", 0) == 0) {  // (a project's file: its bytes, it has no /files/ address)
          std::string bytes;
          std::error_code ec;
          if (std::filesystem::file_size(std::filesystem::path(it.info.path), ec) > kOcrMaxImageBytes || ec) text = "the image is larger than 64 MB";
          else if (files::Store::read(it.info.path, bytes)) ok = ocr_image(rq, rs, "data:" + ocr_image_mime(it.info.name) + ";base64," + apic::base64_encode(bytes), task, tally, text);
          else text = "the file cannot be read";
        } else {
          ok = ocr_image(rq, rs, "/files/" + it.info.id, task, tally, text);
        }
        if (!ok) {
          failed = "file " + std::to_string(k + 1) + (it.pdf ? ", page " + std::to_string(p) : "") + ": " + text;
          text = "[" + failed + "]";
        }
        emit(text);
        if (!ocr_md_add(md, (it.pdf ? "## Page " + std::to_string(p) + " / " + std::to_string(it.pages) + "\n\n" : "") + text + "\n\n") &&
            failed.empty()) {
          failed = "file " + std::to_string(k + 1) + ": could not write " + md.md.filename().string() + " (is the disk full?)";
          emit("[" + failed + "]");
        }
      }
      if (p <= it.pages) break;  // (stopped inside this file: k, p is where the next turn starts)
    }
    if (gone) return true;
    int left = 0, files_left = 0;  // (pages and files not read yet)
    for (int i = k; i < nf; ++i) {
      const int from = i == k ? p : 1;
      if (from <= items[i].pages) left += items[i].pages - from + 1, ++files_left;
    }
    const std::string where = ocr_md_where(md);
    std::string note;
    if (!failed.empty()) {
      note = "\n\n[OCR: stopped at " + failed + (where.empty() ? "" : "; the text before it is in " + where) + "]";
    } else if (left > 0 && !rs.client_gone()) {
      note = "\n\n[OCR: " + std::to_string(left) + " more page(s) in " + std::to_string(files_left) + " file(s); \"lanjut\" reads on from file " +
             std::to_string(k + 1) + " / " + std::to_string(nf) + (items[k].pdf ? ", page " + std::to_string(p) : "") +
             (where.empty() ? "" : ", added to " + where) + "]";
    } else if (saving && fk < nf) {
      if (!per_file) {
        note = "\n\n[OCR: every file is read (" + std::to_string(nf) + "); saved to " + where + "]";
      } else {
        std::string names;
        for (const std::string& m : made) names += (names.empty() ? "" : ", ") + m;
        note = "\n\n[OCR: every file is read (" + std::to_string(nf) + "), each into a .md of its own; this answer's: " + names +
               (dirs.in_chat ? " in this chat's files"
                : " in " + dirs.dir.string() +
                      (!dirs.proj_id.empty() ? " (in the project as project:" + dirs.proj_id + (dirs.proj_rel.empty() ? "" : "/" + dirs.proj_rel) + "/)"
                       : dirs.chat.empty() ? "" : " and this chat's files")) + "]";
      }
    }
    emit(note);
    ocr_finish(rs, stream, id, created, all, tally);
    return true;
  }
  static bool ocr_model(const GgufFile& g, const std::string& id) {
    std::string n = g.get_str("general.name") + " " + id;
    for (char& c : n) c = (char)tolower((unsigned char)c);
    return n.find("ocr") != std::string::npos;
  }
  // pages the user's words name: "halaman 51-100" / "pages 51 to 100" (first, upto), "page 7" (first, upto = first);
  // nothing: both left as they are
  static void ocr_pages_asked(const std::string& w, int& first, int& upto) {
    std::string t;
    for (char c : w) t += (char)tolower((unsigned char)c);
    for (const char* key : {"halaman", "hal.", "hal ", "pages", "page"}) {
      const size_t k = t.find(key);
      if (k == std::string::npos) continue;
      const char* p = t.c_str() + k + strlen(key);
      while (*p == ' ' || *p == ':') ++p;
      if (!isdigit((unsigned char)*p)) continue;
      // (a number of at most 7 digits: atoi of "page 99999999999999999999" overflowed an int, BUGS.md 4b)
      auto num = [](const char*& q) {
        long v = 0;
        int nd = 0;
        for (; isdigit((unsigned char)*q); ++q)
          if (++nd <= 7) v = v * 10 + (*q - '0');
        return nd <= 7 ? (int)v : 0;
      };
      const int a = num(p);
      while (*p == ' ') ++p;
      int b = a;
      if (*p == '-' || !strncmp(p, "sampai", 6) || !strncmp(p, "to", 2) || !strncmp(p, "s/d", 3) || !strncmp(p, "\xE2\x80\x93", 3)) {
        p += *p == '-' ? 1 : !strncmp(p, "sampai", 6) ? 6 : !strncmp(p, "\xE2\x80\x93", 3) ? 3 : !strncmp(p, "s/d", 3) ? 3 : 2;
        while (*p == ' ') ++p;
        if (isdigit((unsigned char)*p)) b = num(p);
      }
      if (a >= 1 && b >= a) {
        first = a;
        upto = std::min(b, a + kOcrMaxPages - 1);
      }
      return;
    }
  }
  // the highest "## Page N" an answer holds with its text after it (0: none). (A heading goes out before its page is
  // read: one stopped there has no text, and that page was skipped by the next "lanjut"; audit 2026-10-31)
  // A heading is a whole line "## Page N / M" (the model's own "## Page 99" in a page's text is not one), and a page
  // whose text is only the error it stopped at ("[page 7: ...]", "[file 2, page 7: ...]") is not read: the next
  // "lanjut" reads it again (self audit 2026-11-08: both moved the start, pages were skipped)
  static int ocr_last_page(const std::string& s) {
    auto head = [&](size_t p) {
      int n = 0, m = 0;
      return (p == 0 || s[p - 1] == '\n') && sscanf(s.c_str() + p, "## Page %d / %d", &n, &m) == 2 ? n : 0;
    };
    auto next_head = [&](size_t from) {
      for (size_t p = s.find("## Page ", from); p != std::string::npos; p = s.find("## Page ", p + 1))
        if (head(p) > 0) return p;
      return std::string::npos;
    };
    int best = 0;
    for (size_t p = next_head(0); p != std::string::npos;) {
      const size_t eol = s.find('\n', p), next = next_head(p + 8);
      const size_t stop = next == std::string::npos || next < eol ? s.size() : next;  // (BUGS.md 11: no underflow)
      const std::string body = eol == std::string::npos ? std::string() : s.substr(eol, stop - eol);
      if (ocr_has_text(body)) best = std::max(best, head(p));  // (a kept note, see the web UI's ocrKept, counts)
      p = next;
    }
    return best;
  }
  // text that is a page's or a file's: not blank, and not only the error the reading stopped at
  static bool ocr_has_text(std::string body) {
    for (size_t a; (a = body.find("[OCR:")) != std::string::npos;) {  // (the answer's own notes are no page's text)
      const size_t b = body.find(']', a);
      body.erase(a, b == std::string::npos ? std::string::npos : b - a + 1);
    }
    const size_t a = body.find_first_not_of(" \r\n\t");
    if (a == std::string::npos) return false;
    const size_t b = body.find_last_not_of(" \r\n\t");
    const std::string t = body.substr(a, b - a + 1);
    const bool error = t.back() == ']' && t.find('\n') == std::string::npos && (t.rfind("[page ", 0) == 0 || t.rfind("[file ", 0) == 0);
    return !error;
  }
  // the model's task prompts (its README): name, prompt
  static const std::vector<std::pair<std::string, std::string>>& ocr_tasks() {
    static const std::vector<std::pair<std::string, std::string>> t = {
        {"text", "Extract the text in the image."},
        {"document",
         "Extract all information from the main body of the document image and represent it in markdown format, "
         "ignoring headers and footers. Tables should be expressed in OTSL format, formulas in the document should be "
         "represented using LATEX format, and the parsing should be organized according to the reading order."},
        {"table", "Parse the table in the image into OTSL."},
        {"formula", "Identify the formula in the image and represent it using LATEX format."},
        {"key", "Extract key information in the image"}};
    return t;
  }
  // the task the user's words ask for (Indonesian or English); "text" when they name none
  static std::string ocr_task_of(const std::string& text) {
    std::string t = text;
    for (char& c : t) c = (char)tolower((unsigned char)c);
    auto has = [&](std::initializer_list<const char*> ws) {
      for (const char* w : ws)
        if (t.find(w) != std::string::npos) return true;
      return false;
    };
    if (has({"json", "key information", "informasi kunci", "data penting", "isian", "field"})) return "key";
    if (has({"tabel", "table", "otsl"})) return "table";
    if (has({"rumus", "formula", "latex", "persamaan", "equation"})) return "formula";
    if (has({"dokumen", "document", "markdown", "halaman", "page"})) return "document";
    return "text";
  }
  // The key-information prompt's tail as the model's README writes it: "Please output ... schema:" and a JSON object
  // of empty fields. The user's own {...} as it is; else the fields their words list after a colon or the task's
  // word ("json: total, tanggal dan nama toko" -> total, tanggal, nama_toko). (2026-10-31, the README's receipt:
  // "json: total dan tanggal" given as plain words gave a made-up {"title", "bbox_2d", ...} object)
  static std::string ocr_schema(const std::string& text) {
    const std::string head = "\n\nPlease output the key information in JSON format according to the following schema:\n";
    const size_t ob = text.find('{'), cb = text.rfind('}');
    if (ob != std::string::npos && cb != std::string::npos && cb > ob) return head + text.substr(ob, cb - ob + 1);
    std::string t = text;
    for (char& c : t) c = (char)tolower((unsigned char)c);
    size_t from = t.find(':');
    if (from == std::string::npos)
      for (const char* w : {"json", "key information", "informasi kunci", "data penting", "isian", "field"})
        if (size_t p = t.find(w); p != std::string::npos) { from = p + strlen(w) - 1; break; }
    if (from == std::string::npos) return "";
    std::string list = t.substr(from + 1);
    for (const char* sep : {" dan ", " and ", " serta ", ";", "\n"})
      for (size_t p; (p = list.find(sep)) != std::string::npos;) list.replace(p, strlen(sep), ",");
    std::vector<std::string> keys;
    for (size_t a = 0; a <= list.size();) {
      size_t b = list.find(',', a);
      if (b == std::string::npos) b = list.size();
      std::string k = ocr_key(list.substr(a, b - a));
      if (!k.empty() && k.size() <= 40 && keys.size() < 32) keys.push_back(k);
      a = b + 1;
    }
    if (keys.empty()) return "";
    std::string s = head + "{\n";
    for (size_t i = 0; i < keys.size(); ++i) s += "    \"" + keys[i] + "\": \"\"" + (i + 1 < keys.size() ? ",\n" : "\n");
    return s + "}";
  }
  // A field the user named as the model's English key: its words in English (common Indonesian receipt / form words;
  // the model was trained on English keys: "tanggal" gave a made-up "bual", "date" the receipt's date), "name / address
  // / number of X" as X_name, X_address, X_number, joined by "_" ("nama toko" -> seller_name)
  static std::string ocr_key(const std::string& words) {
    static const std::map<std::string, std::string> en = {
        {"tanggal", "date"}, {"tgl", "date"}, {"waktu", "time"}, {"jam", "time"}, {"jumlah", "amount"},
        {"harga", "price"}, {"pajak", "tax"}, {"ppn", "tax"}, {"nama", "name"}, {"toko", "seller"}, {"penjual", "seller"},
        {"pembeli", "buyer"}, {"pelanggan", "customer"}, {"alamat", "address"}, {"nomor", "number"}, {"no", "number"},
        {"telepon", "phone"}, {"telp", "phone"}, {"hp", "phone"}, {"faktur", "invoice"}, {"barang", "item"},
        {"diskon", "discount"}, {"potongan", "discount"}, {"kembalian", "change"}, {"tunai", "cash"},
        {"bayar", "payment"}, {"pembayaran", "payment"}, {"perusahaan", "company"}, {"penerima", "recipient"},
        {"pengirim", "sender"}, {"tujuan", "destination"}, {"keterangan", "description"}, {"total", "total"}};
    std::vector<std::string> w;
    std::string cur;
    for (char c : words + " ") {
      if (isalnum((unsigned char)c)) { cur += c; continue; }
      if (!cur.empty()) {
        auto it = en.find(cur);
        w.push_back(it != en.end() ? it->second : cur);
        cur.clear();
      }
    }
    if (w.size() >= 2 && (w[0] == "name" || w[0] == "address" || w[0] == "number" || w[0] == "date")) {
      std::rotate(w.begin(), w.begin() + 1, w.end());  // ("nama toko": seller_name)
    }
    std::string k;
    for (const std::string& x : w) k += (k.empty() ? "" : "_") + x;
    return k;
  }
  // The answer's OTSL tables (<fcel> a cell, <ecel> empty, <lcel> / <ucel> / <xcel> merged into the cell left / above /
  // both, <nl> a row's end) as Markdown tables, the first row as the header, merged cells empty: what a client shows
  // as a table (the web UI does the same while an answer streams, webui/index.html otslToMd). Text around them as is.
  static std::string otsl_to_md(const std::string& s) {
    if (s.find("<fcel>") == std::string::npos && s.find("<ecel>") == std::string::npos && s.find("<ched>") == std::string::npos)
      return s;
    static const char* tags[] = {"fcel", "ecel", "lcel", "ucel", "xcel", "ched", "rhed", "srow", "nl"};
    auto tag_at = [&](size_t p) -> std::string {  // the OTSL tag starting at p, or ""
      if (s[p] != '<') return "";
      for (const char* t : tags) {
        const std::string x = std::string("<") + t + ">";
        if (s.compare(p, x.size(), x) == 0) return t;
      }
      return "";
    };
    auto esc = [](std::string c) {
      std::string o;
      for (char ch : c) o += ch == '|' ? std::string("\\|") : ch == '\n' ? std::string(" ") : std::string(1, ch);
      size_t a = o.find_first_not_of(' '), b = o.find_last_not_of(' ');
      return a == std::string::npos ? std::string() : o.substr(a, b - a + 1);
    };
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
      size_t start = i;
      if (s.compare(i, 6, "<otsl>") == 0) i += 6;
      if (tag_at(i).empty()) {  // (plain text up to the next tag)
        i = start;
        const size_t n = s.find('<', i + 1);
        out += s.substr(i, n == std::string::npos ? std::string::npos : n - i);
        i = n == std::string::npos ? s.size() : n;
        continue;
      }
      std::vector<std::vector<std::string>> grid(1);
      while (i < s.size()) {
        const std::string t = tag_at(i);
        if (t.empty()) break;
        i += t.size() + 2;
        if (t == "nl") {
          grid.emplace_back();
          while (i < s.size() && (s[i] == '\n' || s[i] == ' ' || s[i] == '\r')) ++i;
          continue;
        }
        size_t e = i;  // (the cell's text: up to the next tag or </otsl>)
        for (;; ++e) {
          e = s.find('<', e);
          if (e == std::string::npos) { e = s.size(); break; }
          if (!tag_at(e).empty() || s.compare(e, 7, "</otsl>") == 0) break;
        }
        const bool text = t == "fcel" || t == "ched" || t == "rhed" || t == "srow";
        grid.back().push_back(text ? esc(s.substr(i, e - i)) : std::string());
        i = e;
      }
      if (s.compare(i, 7, "</otsl>") == 0) i += 7;
      while (!grid.empty() && grid.back().empty()) grid.pop_back();
      size_t w = 0;
      for (const auto& r : grid) w = std::max(w, r.size());
      if (!w) continue;
      auto row = [&](const std::vector<std::string>& r) {
        std::string l = "|";
        for (size_t c = 0; c < w; ++c) l += " " + (c < r.size() ? r[c] : std::string()) + " |";
        return l + "\n";
      };
      out += "\n\n" + row(grid[0]) + "|";
      for (size_t c = 0; c < w; ++c) out += " --- |";
      out += "\n";
      for (size_t r = 1; r < grid.size(); ++r) out += row(grid[r]);
      out += "\n";
    }
    return out;
  }
  Json ocr_request(const Json& in) {
    const Json& msgs = in["messages"];
    int last = -1;
    for (int i = 0; i < (int)msgs.size(); ++i)
      if (msgs[i]["role"].str() == "user") last = i;
    if (last < 0) return in;
    // the last user turn's words; its images, else those of the latest earlier turn that had one (a follow-up in a
    // chat, "now as a table": the same image again, with the new task)
    Json images = Json::array();
    std::string text;
    if (msgs[last]["content"].is_string()) text = msgs[last]["content"].str();
    // (only a turn of words alone looks back: one that brought a file of another kind is about that file)
    bool other_file = false;
    for (const Json& p : msgs[last]["content"].items())
      other_file = other_file || p["text"].str().rfind("[file: ", 0) == 0 || p["type"].str() == "input_file";
    for (int i = last; i >= 0 && !images.size() && !(i < last && other_file); --i) {
      if (msgs[i]["role"].str() != "user") continue;
      for (const Json& p : msgs[i]["content"].items()) {
        const std::string ty = p["type"].str();
        if (ty == "image_url" || ty == "input_image" || ty == "image") images.push(p);
        else if (i == last && (ty == "text" || ty == "input_text") && p["text"].str().rfind("[file: ", 0) != 0)
          text += (text.empty() ? "" : "\n") + p["text"].str();  // (not a saved file's line)
      }
    }
    if (!images.size()) return in;  // (no image in the conversation: the request as it came)
    std::string prompt;
    for (const auto& [name, pr] : ocr_tasks())
      if (text.rfind(pr, 0) == 0) prompt = text;  // (an official prompt already, with a schema after it for key)
    if (prompt.empty()) {
      const std::string task = ocr_task_of(text);
      for (const auto& [name, pr] : ocr_tasks())
        if (name == task) prompt = pr;
      if (task == "key") prompt += ocr_schema(text);
    }
    Json content = images, txt = Json::object(), msg = Json::object(), out = in, kw = Json::object();
    txt.set("type", "text");
    txt.set("text", prompt);
    content.push(txt);
    msg.set("role", "user");
    msg.set("content", content);
    Json one = Json::array();
    one.push(msg);
    out.set("messages", one);
    out.set("tools", Json());
    out.set("tool_choice", Json());
    out.set("enable_thinking", false);
    if (in["chat_template_kwargs"].is_object()) kw = in["chat_template_kwargs"];
    kw.set("enable_thinking", false);
    out.set("chat_template_kwargs", kw);
    out.set("reasoning_effort", Json());
    out.set("temperature", 0.0);
    out.set("max_tokens", std::max(4096.0, in["max_tokens"].as_number(0)));
    zlog::eprintf("OCR mode: %d image(s), task prompt \"%.60s%s\"\n", (int)images.size(), prompt.c_str(),
            prompt.size() > 60 ? "..." : "");
    return out;
  }

 public:
