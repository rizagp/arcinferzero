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
  static constexpr int kOcrMaxPages = 50;  // (a PDF's pages read per request; more: a note at the end)
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
  //   of this computer, for a request from this computer only (default: the chat's folder, its Files panel). A new
  //   file never takes an existing one's place: "name (2).md" then; append (the web UI's "lanjut" turns): the named
  //   file goes on. Every answer ends with the file it went to ("[OCR: ... added to F]" / "[OCR: saved to F]").
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
    auto pdf_of = [](const Json& m, files::Info& out) {
      for (const Json& p : m["content"].items()) {
        if (p["type"].str() != "file" && p["type"].str() != "input_file") continue;
        files::Info i;
        const std::string fid = p["file"]["file_id"].str(p["file_id"].str());
        if (files::store().get(fid, i) && files::lower(i.name).size() > 4 &&
            files::lower(i.name).rfind(".pdf") == files::lower(i.name).size() - 4) {
          out = i;
          // (the name the chat gave it: one file in several chats is one entry under whichever name came first)
          if (p["file"]["filename"].is_string() && !p["file"]["filename"].str().empty()) out.name = p["file"]["filename"].str();
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
    const std::string words = text_of(msgs[last]), task = text_of(msgs[at]);  // (a continuation keeps the PDF's task)
    std::string err;
    const int pages = pdfr::pdf_page_count(pdf.path, err);
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
    if (stream) rs.sse_begin();
    std::string all;
    int64_t ptok = 0, ctok = 0;
    // the .md next to the chat's files (started over with page 1, else appended)
    std::filesystem::path md;
    bool in_chat = false;  // (in the chat's folder: its Files panel follows the file)
    const Json& sv = body["ocr_save"];
    if (sv.as_bool(false) || sv.is_object()) {
      std::string name = sv["name"].str();
      if (name.empty()) name = std::filesystem::path(pdf.name).stem().string() + ".md";
      name = std::filesystem::path(name).filename().string();  // (a name, never a path)
      for (char& c : name)
        if (strchr("<>:\"/\\|?*", c) || (unsigned char)c < 32) c = '_';
      if (name.empty() || name[0] == '.') name = "ocr" + name;
      if (files::lower(name).size() < 3 || files::lower(name).rfind(".md") != files::lower(name).size() - 3) name += ".md";
      std::filesystem::path dir;
      const std::string folder = sv["folder"].str();
      std::error_code ec;
      if (!folder.empty()) {
        if (!rq.local_peer) {
          rs.send(403, "application/json", "{\"error\":{\"message\":\"ocr_save.folder: from this computer only\"}}");
          return true;
        }
        dir = std::filesystem::path(folder);
        if (!dir.is_absolute() || !std::filesystem::is_directory(dir, ec)) {
          rs.send(400, "application/json", "{\"error\":{\"message\":" + Json("no such folder: " + folder).dump() + "}}");
          return true;
        }
      } else {  // (the asking chat's folder: the PDF's copy found may be another chat's)
        const std::string chat = body["chat_id"].str();
        dir = files::valid_id(chat) ? std::filesystem::path(files::store().chat_dir_make(chat)) : std::filesystem::path(pdf.path).parent_path();
        in_chat = true;
      }
      md = dir / name;
      if (!sv["append"].as_bool(false)) {  // (a new file: never in an existing one's place)
        const std::string stem = md.stem().string();
        for (int k = 2; std::filesystem::exists(md, ec) && k < 1000; ++k) md = dir / (stem + " (" + std::to_string(k) + ").md");
        std::ofstream(md, std::ios::binary | std::ios::trunc) << "# " << pdf.name << "\n\n";
      }
    }
    auto save = [&](const std::string& piece) {
      if (md.empty()) return;
      std::ofstream(md, std::ios::binary | std::ios::app) << piece;
      if (in_chat) files::store().refresh();  // (in the Files panel with its new size)
    };
    const std::string where = md.empty() ? "" : in_chat ? md.filename().string() : md.string();
    fprintf(stderr, "OCR mode: %s, pages %d-%d of %d%s\n", pdf.name.c_str(), first, n, pages,
            md.empty() ? "" : (", saved to " + md.filename().string()).c_str());  // (the file's name only: no user's path in the log)
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
    for (int p = first; p <= n && !rs.client_gone(); ++p) {
      std::string piece = (p > first ? "\n\n" : "") + std::string("## Page ") + std::to_string(p) + " / " + std::to_string(pages) + "\n\n";
      const std::string png = pdfr::pdf_render_page(pdf.path, p, 150, err);
      if (png.empty()) {
        piece += "[the page could not be rendered: " + err + "]";
      } else {
        Json url = Json::object(), img = Json::object(), txt = Json::object(), content = Json::array(), msg = Json::object(),
             one = Json::array(), req = Json::object();
        url.set("url", "data:image/png;base64," + apic::base64_encode(png));
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
        chat(rq2, r2);  // (an image now: ocr_request gives it the task prompt; its tables come back as Markdown)
        Json j;
        Json::try_parse(out, j);
        if (st != 200) piece += "[page " + std::to_string(p) + ": " + j["error"]["message"].str("HTTP " + std::to_string(st)) + "]";
        else piece += j["choices"][0]["message"]["content"].str();
        ptok += j["usage"]["prompt_tokens"].as_int();
        ctok += j["usage"]["completion_tokens"].as_int();
      }
      save((p > 1 && p == first ? "\n\n" : "") + piece);
      if (stream) {
        Json d = Json::object();
        d.set("content", piece);
        if (!rs.sse_event(chunk(id, created, d, Json()).dump())) return true;
      } else {
        all += piece;
      }
    }
    std::string note;  // (pages left: the web UI's cue for "lanjut"; and the file the text went to)
    if (pages > n && n >= first) {
      const int a = n + 1, b = std::min(pages, n + kOcrMaxPages);
      note = "\n\n[OCR: " + std::to_string(pages - n) + " more page(s); \"lanjut\" reads pages " + std::to_string(a) + "-" +
             std::to_string(b) + (md.empty() ? "" : ", added to " + where) + "]";
    } else if (!md.empty()) {
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
    Json usage = Json::object();
    usage.set("prompt_tokens", ptok);
    usage.set("completion_tokens", ctok);
    usage.set("total_tokens", ptok + ctok);
    if (stream) {
      Json fin = chunk(id, created, Json::object(), Json("stop"));
      fin.set("usage", usage);
      rs.sse_event(fin.dump());
      rs.sse_event("[DONE]");
      rs.sse_end();
      return true;
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
    rs.send(200, "application/json", r.dump());
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
      const int a = atoi(p);
      while (isdigit((unsigned char)*p)) ++p;
      while (*p == ' ') ++p;
      int b = a;
      if (*p == '-' || !strncmp(p, "sampai", 6) || !strncmp(p, "to", 2) || !strncmp(p, "s/d", 3) || !strncmp(p, "\xE2\x80\x93", 3)) {
        p += *p == '-' ? 1 : !strncmp(p, "sampai", 6) ? 6 : !strncmp(p, "\xE2\x80\x93", 3) ? 3 : !strncmp(p, "s/d", 3) ? 3 : 2;
        while (*p == ' ') ++p;
        if (isdigit((unsigned char)*p)) b = atoi(p);
      }
      if (a >= 1 && b >= a) {
        first = a;
        upto = std::min(b, a + kOcrMaxPages - 1);
      }
      return;
    }
  }
  // the highest "## Page N" an answer holds (0: none)
  static int ocr_last_page(const std::string& s) {
    int best = 0;
    for (size_t p = s.find("## Page "); p != std::string::npos; p = s.find("## Page ", p + 8)) best = std::max(best, atoi(s.c_str() + p + 8));
    return best;
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
    fprintf(stderr, "OCR mode: %d image(s), task prompt \"%.60s%s\"\n", (int)images.size(), prompt.c_str(),
            prompt.size() > 60 ? "..." : "");
    return out;
  }

 public:
