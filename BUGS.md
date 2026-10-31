# Potential bugs found

**Scope:** static review of the repository, plus an MSVC `/W4` syntax compile and `/analyze:only` pass. I did **not** start the server.
**Date:** 2026-10-31

This is a list of likely defects and hardening issues, not a proof that each is exploitable. I have marked confidence where the issue depends on runtime conditions.

## Summary

| # | Severity | Confidence | Area | Issue |
|---|----------|------------|------|-------|
| 1 | High/Medium | Medium (preconditions) | `src/http.h`, `tools/server_ocr.h` | A local page from another port can call `/v1/chat/completions`; `ocr_save.folder` only checks `local_peer`, so it can write `.md` files outside the data directory. |
| 2 | Medium | High | `tools/server_ocr.h` | OCR save collision loop can overwrite an existing file after 999 name collisions. |
| 3 | Medium | High | `tools/server_files.h` | `chat.json.tmp` (and case variants of `chat.json`) are not reserved; a user file can be silently overwritten by the chat saver. |
| 4 | Medium | High | `tools/server.cpp`, `tools/server_ocr.h`, `tools/server_whisper.h` | Untrusted JSON numbers are cast with C-style casts / `atoi` before range checks; out-of-range values are undefined behaviour. |
| 5 | Medium | High | `tools/server_files.h` | `refs()` only recognises compact JSON (`"file_id":"..."`); pretty-printed chat JSON loses file references and can lead to deleted files. |
| 6 | Low | High | `tools/server_ocr.h` | Saving explicitly into the chat's own folder leaves `in_chat == false`, so the answer reports a full path and skips the store refresh. |
| 7 | Low | High | `src/http.h` | `reply_close()` sends `Connection: keep-alive` via `send()` and then closes the socket. |
| 8 | Low | High | `zero/engine.h` | `ARC_QWEN_PS` can divide by zero when `phg` or `ptt` is 0. |
| 9 | Low | High | `tools/server.cpp` | `read_file` can return text starting in the middle of a UTF-8 sequence when a client passes an arbitrary `offset`. |
| 10 | Low | High | `webui/index.html` | SSE event payloads are passed to `JSON.parse()` without a guard. |
| 11 | Low | Medium | `tools/server_ocr.h` | `ocr_last_page()` can underflow a length if a second page heading occurs before the first newline. |
| 12 | Info | — | MSVC `/analyze` | Several warnings worth hardening or confirming (see appendix). |

---

## 1. Cross-origin local page can reach OCR file saving

**Files:** `src/http.h:330-347`, `tools/server_ocr.h:118-127`

`foreign_ok()` deliberately allows a page served by another local program to POST to `/v1/chat/completions`:

```cpp
// src/http.h:341-347
static const char* const gen[] = {"/v1/chat/completions", ...};
for (const char* g : gen)
  if (p == g) return true;
```

But the chat handler contains a server-side file-write feature, `ocr_save`:

```cpp
// tools/server_ocr.h:123-127
if (!folder.empty()) {
  if (!rq.local_peer) {
    rs.send(403, ..., "ocr_save.folder: from this computer only");
    return true;
  }
  dir = std::filesystem::path(folder);
  ...
```

`rq.local_peer` only says that the TCP connection came from `127.0.0.1`/`::1`; it does **not** say the request came from the server's own web UI. A page from `http://localhost:<another-port>/` is also `local_peer == true`.

So a browser page served by another local program can POST a chat request with `ocr_save: {folder: "C:\\Users\\...\\somewhere", name: "x", append: true}`. If the request also references a known PDF in the file store and an OCR model is loaded, the server will append model-generated text to `somewhere\x.md` or create it.

**Impact:** breaks the intended local-origin trust boundary ("pages of other local servers: the model API only") and gives a local web page a server-side file-write primitive. With `append: true` it can corrupt an existing `.md` file.

**Preconditions:** an OCR model is loaded, the request has a PDF `file_id` known to the page, and the target folder exists. The first condition is realistic for OCR users; the second limits practical exploitability, because `/v1/files` and `/files` are blocked for foreign origins. Still, the check is on the wrong trust property.

**Suggested fix:**
- Check `same_origin(origin, host)` or a dedicated local-only origin flag for `ocr_save.folder`, not only `local_peer`.
- Or move OCR saving to a separate endpoint under `/v1/tools`/`/v1/server` that `foreign_ok()` already blocks.
- Consider forbidding `append: true` for arbitrary existing paths, or requiring same-origin confirmation.

---

## 2. OCR save can overwrite after 999 name collisions

**File:** `tools/server_ocr.h:143-158`

```cpp
const std::string stem = md.stem().string();
for (int k = 2; (exists(md) || (!md_chat.empty() && exists(md_chat))) && k < 1000; ++k) {
  name = stem + " (" + std::to_string(k) + ").md";
  md = dir / name;
  ...
}
for (const auto& f_path : {md, md_chat}) {
  ...
  std::ofstream f(f_path, std::ios::binary | std::ios::trunc);
```

The loop stops when `k == 1000` even if `md` is still occupied. The following code then opens `md` with `std::ios::trunc`, overwriting the existing file.

**Trigger:** the target folder already contains `stem.md`, `stem (2).md`, ..., `stem (999).md` (or the chat copy variants). This is unlikely by accident but violates the documented "never over an existing file" rule and is easy to force in a folder the caller controls.

**Fix:** after the loop, re-check `exists(md)` / `exists(md_chat)` and fail with a clear error, or keep searching (with a sane cap and a randomized suffix).

---

## 3. `chat.json.tmp` and case variants of `chat.json` are not reserved

**Files:** `tools/server_files.h:610-620`, `tools/server_files.h:698-700`, `tools/server_files.h:421-437` (`chat_put`)

`free_name()` reserves only the exact spelling `chat.json`:

```cpp
if (name == "chat.json" || fs::exists(dir / name, ec)) { ... }
```

`scan()` deliberately ignores the chat-write temp file:

```cpp
if (!f.is_regular_file() || fn.rfind(".", 0) == 0 || fn == "chat.json" || fn == "chat.json.tmp") continue;
```

But `chat_put()` uses exactly that name as its temp file:

```cpp
const fs::path p = dir / "chat.json", tmp = dir / "chat.json.tmp";
write(tmp, body);
fs::rename(tmp, p, ec);
```

A file created through `create_file` (or an upload) named `chat.json.tmp` is therefore:
- hidden from the Files panel (`scan()` skips it),
- overwritten by the next chat save,
- and never linked/listed as a normal user file.

Additionally, the comparison is case-sensitive. On Windows, a user file named `CHAT.JSON` in a new chat folder is not reserved; when the chat is saved, `chat.json` refers to the same case-insensitive path and can replace it.

**Fix:**
- Reserve `chat.json`, `chat.json.tmp` and all case variants.
- Better: write temp files into a private temp subdirectory (or use a random suffix not reachable by `safe_name`), and make reservation checks case-insensitive on Windows.

---

## 4. Untrusted numbers cast with `atoi` / C-style casts before validation

### 4a. `read_file` (`tools/server.cpp:1410`, `1441-1442`)

```cpp
// source-file branch
const size_t off = std::min(body.size(), (size_t)std::max(0.0, a["offset"].as_number(0)));

// pages branch
const std::string pg = a["pages"].is_number()
    ? std::to_string((int)a["pages"].as_number())
    : a["pages"].str();
const int from = std::max(1, atoi(pg.c_str())), ...
```

`Json::as_number()` accepts any finite JSON number, including `1e308`. Casting a `double` outside the destination integer range is undefined behaviour in C++. `atoi()` is also undefined for values outside `int`. These values are reachable through the public `/v1/tools/read_file` endpoint.

**Fix:** use `as_clamped(0, 0, (int64_t)body.size())` for `offset` and `as_clamped(...)` or `std::from_chars` for `pages`/page ranges; reject huge digit strings before parsing.

### 4b. OCR page numbers (`tools/server_ocr.h:320-327`)

```cpp
const int a = atoi(p);
...
if (isdigit((unsigned char)*p)) b = atoi(p);
```

The text comes from the user prompt. A string such as `page 999999999999999999999` invokes the same integer-overflow issue. Clamp/validate with a maximum page number.

### 4c. Transcript export query parameters (`tools/server_whisper.h:774-776`)

```cpp
co.max_chars = std::clamp(atoi(...), 8, 200);
co.max_lines = std::clamp(atoi(...), 1, 4);
co.max_s = std::clamp(atof(...), 0.5, 30.0);
```

`atoi`/`atof` accept arbitrarily long strings; `atoi` overflow is undefined and `atof` may produce `inf` before the clamp. Use `from_chars`/`strtod` with error handling.

**Impact:** typically implementation-defined results rather than a crash, but it is still UB and can produce nonsensical page ranges or limits. Under `/fp:fast` and other optimizers, behaviour is not guaranteed.

---

## 5. Chat file references are found only in compact JSON

**File:** `tools/server_files.h:482-500`

```cpp
for (const char* pat : {"/files/", "\"file_id\":\""})
  for (size_t i = 0, n = strlen(pat); (i = text.find(pat, i)) != std::string::npos; i += n) {
    std::string fid = text.substr(i + n, 21);
...
```

The web UI writes compact JSON, so its references are found. Any client that pretty-prints the same chat JSON (`"file_id": "abc..."` with a space) will not have its file references recognised.

The consequences:
- `chat_put()` will not `link()` those files into the chat's folder.
- `uses()` will not list the chat as a user of the file.
- `/v1/files/delete_unused` can delete a file that is still referenced by a saved pretty-printed chat.

**Fix:** parse the chat JSON with `Json::try_parse` and walk `messages`/parts for `file_id` values; keep the string scan only as a fallback.

---

## 6. OCR save into the chat's own folder is reported incorrectly

**File:** `tools/server_ocr.h:133-170`

```cpp
if (std::filesystem::equivalent(dir, chat_dir, ec)) chat_dir.clear();  // the chat's own folder named
...
auto save = [&](const std::string& piece) {
  ...
  if (in_chat || !md_chat.empty()) files::store().refresh();
};
...
const std::string where = md.empty() ? "" : in_chat ? md.filename().string()
                                     : md.string() + ...;
```

When the user explicitly passes the chat folder as `ocr_save.folder`, `chat_dir` is cleared but `in_chat` remains `false`. The file is still written to the chat folder, but:
- the final answer reports the full absolute path instead of the file name,
- `files::store().refresh()` is skipped until the next scan.

**Fix:** set `in_chat = true` when `equivalent(dir, chat_dir)` succeeds, and clear the `md_chat` copy as now.

---

## 7. Error replies advertise keep-alive, then close the socket

**File:** `src/http.h:359-363`

```cpp
static void reply_close(SOCKET c, int status, const char* msg) {
  HttpResponse rs(c);
  rs.send(status, "application/json", ...);  // send() emits "Connection: keep-alive"
  closesocket(c);
}
```

Clients may cache the connection and reuse it for the next request, only to find it was closed. This is mostly a protocol-correctness issue for malformed requests and 403/413/503 replies.

**Fix:** add a `Connection: close` mode to `HttpResponse::send()` (or a boolean) for replies that are followed by a close.

---

## 8. `ARC_QWEN_PS` can divide by zero

**File:** `zero/engine.h:1996-2003`

```cpp
PsCfg c;
if (hp.gemma || hd != 256) return c;
if (const char* e = getenv("ARC_QWEN_PS")) {
  sscanf(e, "%d,%d,%d,%d,%d", &c.pns, &c.phg, &c.ptt, &c.tpw, &c.from);
  if (c.pns && (hpg % c.phg || c.tpw % c.ptt)) c.pns = 0;
  return c;
}
```

Missing fields leave `phg`/`ptt` at zero. With `ARC_QWEN_PS=1,0,...` or `ARC_QWEN_PS=1,2,0,...`, the `%` operations divide by zero (SIGFPE on x86).

This is environment-only, not remotely reachable, but it crashes the process at model load. The ignored `sscanf` return also means partially specified values are accepted silently.

**Fix:** validate all parsed fields before the modulo, and reject non-positive values.

---

## 9. `read_file` can start in the middle of a UTF-8 character

**File:** `tools/server.cpp:1559-1567`

```cpp
const size_t off = (size_t)a["offset"].as_clamped(0, 0, (int64_t)data.size());
...
while (end < data.size() && end > off && ((unsigned char)data[end] & 0xC0) == 0x80) --end;
r.set("content", off < data.size() ? data.substr(off, end - off) : std::string());
```

The cut at `end` is moved back to a character boundary, but `off` is not. A client or model passing an arbitrary offset can receive a string starting with UTF-8 continuation bytes. `Json::dump()` does not validate UTF-8, so the response can contain invalid UTF-8.

The safe path is to move `off` forward to the next non-continuation byte, or reject an offset inside a character. Several other branches in the same file already do this adjustment; this one missed it.

---

## 10. Web UI trusts every SSE `data:` line as JSON

**File:** `webui/index.html:1924-1927`

```js
if (!ev.startsWith("data:")) continue;
const data = ev.slice(5).trim();
if (data === "[DONE]") continue;
const j = JSON.parse(data);
```

The server currently sends only JSON or `[DONE]`, but an intermediary, proxy or future server event that emits a non-JSON `data:` payload will throw out of the read loop. The answer then stops with a JSON syntax error instead of ignoring the unknown event.

**Fix:** wrap the parse in `try/catch` and skip malformed events, as the code already does for tool arguments.

---

## 11. `ocr_last_page()` length underflow on malformed output

**File:** `tools/server_ocr.h:339-348`

```cpp
const size_t eol = s.find('\n', p), next = s.find("## Page ", p + 8);
const std::string body = eol == std::string::npos ? std::string()
    : s.substr(eol, (next == std::string::npos ? s.size() : next) - eol);
```

If the next `"## Page "` occurs before the end of the heading line (`next < eol`), `next - eol` underflows `size_t` and `substr` takes the rest of the string. The current page is then counted as having text because a later page's text is included.

The normal server output puts a newline after the heading, so this needs malformed/non-web-UI output to trigger. Still, clamp the length:

```cpp
const size_t stop = next == std::string::npos || next < eol ? s.size() : next;
```

---

## Appendix: MSVC `/analyze` warnings

These were produced by `cl /analyze:only` over `zero/serverz.cpp`, `zero/runz.cpp`, `zero/transcribez.cpp`, `tools/arc_recorder.cpp`, `zero/vision_testz.cpp` and `zero/asr_testz.cpp`. Some are likely false positives, but they are worth checking.

| File | Warning | Note |
|------|---------|------|
| `zero/expert_store.h:56` | C6001 uninitialized `slots_.ev` | `Slot` has `HANDLE ev = nullptr`, so this may be a false positive; still, the destructor loops over the raw array before any explicit initialization, so an explicit `for (auto& s : slots_) s.ev = nullptr;` in the constructor would remove all doubt. |
| `zero/npu/npu_rt.h:94` | C6387 `GetModuleHandleA("ze_loader.dll")` may be 0 | `GetProcAddress(NULL, ...)` is passed directly. Guard the module handle and return the existing "no NPU device" error before calling `GetProcAddress`. |
| `tools/api_compat.h:795-797` | C6011 possible NULL dereference of `c` | `item()` dereferences `*c` for any type that is not `"reasoning"`/`"message"`. Current call sites pass a `Call*` only for `"function_call"`, so this is latent; add an explicit branch/assert for unknown types. |
| `tools/server_cache.h:129` | C6282 "incorrect operator" | `return !(other = true);` is semantically valid but obscure. Use `other = true; return false;`. |
| `tools/pdf_images.h:274` | C6001 uninitialized `v` | `int v[4]` is filled only for a colour space's component count (1/3/4) and the 4-component branch reads all four; likely a false positive, but initializing `int v[4] = {};` costs nothing and makes the invariant explicit. |
| `tools/doc_writer.h:732,1145` | C6336 operator precedence | `last_num = b.num ? b.num : last_num + 1;` parses as intended; parenthesize the `+ 1` to silence the warning. |
| `zero/engine.h:2001` | C6031 ignored `sscanf` | Related to finding 8: check the parsed field count. |
| `tools/web_tools.h:107`, `tools/arc_recorder.cpp:357` | C6031 ignored `WSAStartup` | A failed WSA startup is not reported until a later network call; check the result and return a clear error. |
| `src/http.h:374`, `tools/web_tools.h:134` | C6262 large stack frames (~66 KB / ~42 KB) | Fine on the default Windows thread stack, but worth knowing for many concurrent connection threads. |

---

## Things checked and not found to be bugs

- The duplicate-looking `for (int i = 1; i < argc; ++i)` around `--list-models` was an artifact of overlapping `sed` ranges; the file has one loop.
- The JSON parser rejects non-finite numbers and deep nesting; the main server paths use `as_clamped` in many places.
- Path IDs for `/files/<id>`, `/v1/chats/<id>` and transcript IDs are validated and contain no path separators.
- The HTTP parser rejects `Transfer-Encoding` + `Content-Length` together, non-chunked transfer encodings, conflicting `Content-Length` values and oversized chunked bodies.
- `foreign_ok()` correctly blocks `/v1/files`, `/files`, `/v1/chats`, `/v1/transcripts`, `/v1/live`, `/v1/tools`, `/v1/server` and `/sessions` for foreign local origins.
- The JSON constraint and tool-call constraint paths are guarded against out-of-range tools/arguments in the reviewed code.

---

## Status (2026-10-31, after the review)

Every finding was checked in the code before it was changed; all eleven were real.

| # | Status | What changed | Checked |
|---|---|---|---|
| 1 | fixed | `ocr_save.folder` only from the server's own page (Origin = http://<Host>) or a client that sends no Origin, besides `local_peer` (`tools/server_ocr.h`) | OCR server: Origin `http://localhost:9999` -> 403; the own page and no Origin -> written |
| 2 | fixed | after the name search, a name still taken -> 409 "choose another name", never a truncating write | read through |
| 3 | fixed | `free_name` reserves `chat.json` and `chat.json.tmp` case-blind (`tools/server_files.h`) | read through |
| 4a | fixed | `read_file`: `offset` by `as_clamped`; `pages` a number by `as_clamped`, a string by a 7-digit reader (`tools/server.cpp`) | read through |
| 4b | fixed | OCR page numbers from the prompt: a 7-digit reader, no `atoi` | read through |
| 4c | fixed | transcript export: `strtol` / `strtod` clamped, NaN keeps the default (`tools/server_whisper.h`) | read through |
| 5 | fixed | `refs()` finds `"file_id"` with whitespace around the colon | read through |
| 6 | fixed | the chat's own folder named as `folder`: `in_chat` set (the name in the answer, the Files panel refreshed) | read through |
| 7 | fixed | `HttpResponse::close_after`: `reply_close` says `Connection: close` (`src/http.h`) | read through |
| 8 | fixed | `ARC_QWEN_PS`: four fields, all positive, else off (`zero/engine.h`) | read through |
| 9 | fixed | `read_file` moves `offset` to a character's start | read through |
| 10 | fixed | the web UI skips a `data:` event that is not JSON | read through |
| 11 | fixed | `ocr_last_page` clamps the length when the next heading comes before the line's end | read through |
| appendix | partly | `npu_rt.h`: the loader's module handle checked before `GetProcAddress`; `pdf_images.h`: `int v[4] = {}`. Left as they are: the `/analyze` notes on `expert_store.h` (the member is initialized), `api_compat.h` (no call site passes a null `Call*`), `server_cache.h` / `doc_writer.h` (style), `WSAStartup` (a later call reports it), the stack frames | - |

Build: serverz, runz, transcribez; `test_api_compat` all passed, `tool_stream_test` all OK.
