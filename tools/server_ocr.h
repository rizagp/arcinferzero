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
  static bool ocr_model(const GgufFile& g, const std::string& id) {
    std::string n = g.get_str("general.name") + " " + id;
    for (char& c : n) c = (char)tolower((unsigned char)c);
    return n.find("ocr") != std::string::npos;
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
    for (int i = last; i >= 0 && !images.size(); --i) {
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
