// CPU-only checks of the web UI's chat folders (tools/server_files.h) and its terminal tool (tools/server_terminal.h),
// in a scratch data folder under %TEMP% (deleted at the end): files into a chat's folder, the folder named at the first
// save, hard links between chats, a file put in or deleted by hand, deleting a chat; a document in a message whole or
// as its outline by the model's context; a command's output, exit code, timeout (with the processes it started) and
// long output; the chat's terminal session (folder, environment, a venv carried from one command to the next).
#include <cstdlib>
#include <fstream>
#include <string>
#include "../tools/server_files.h"
#include "../tools/server_terminal.h"
#include <cstdio>

static int fails = 0;
static void check(const char* name, bool ok, const std::string& got = "") {
  printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name, ok || got.empty() ? "" : "  got: ", ok ? "" : got.c_str());
  fails += !ok;
}

int main() {
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / ("arcinferzero_store_test_" + std::to_string(GetCurrentProcessId()));
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root, ec);
  _putenv_s("ARC_DATA_DIR", root.string().c_str());
  {
    files::Store st;
    // an upload before the chat's first save: a folder with only the date and id
    const files::Info a = st.put("notes.txt", "hello", "c1");
    check("upload into the chat's folder", fs::path(a.path).parent_path().filename().string().find("(c1)") != std::string::npos &&
                                              fs::path(a.path).filename() == "notes.txt", a.path);
    // the first save names the folder from its title; the file follows
    st.chat_put("c1", "{\"title\":\"Rapat mingguan: tim A\",\"history\":[{\"content\":\"/files/" + a.id + "\"}]}");
    const std::string d1 = st.chat_dir("c1");
    check("folder named at the first save", fs::path(d1).filename().string().find(" Rapat mingguan tim A (c1)") == 10, d1);
    // a first save that could not name the folder (here: no title yet) names it at a later save
    st.put("x.bin", "draft", "c9");
    st.chat_put("c9", "{\"title\":\"New chat\",\"history\":[]}");
    const std::string d9 = st.chat_dir("c9");
    st.chat_put("c9", "{\"title\":\"Judul kemudian\",\"history\":[]}");
    {  // (the terminal session's paths move with a renamed folder)
      st.put("y.bin", "draft2", "c8");
      const std::string d8 = st.chat_dir("c8");
      std::string esc;
      for (char ch : d8) esc += ch == '\\' ? std::string("\\\\") : std::string(1, ch);
      { std::ofstream o(fs::path(d8) / ".terminal.json", std::ios::binary); o << "{\"cwd\":\"" << esc << "\",\"env\":{\"VIRTUAL_ENV\":\"" << esc << "\\\\.venv\"}}"; }
      st.chat_put("c8", "{\"title\":\"Venv pindah\",\"history\":[]}");
      std::string t8, esc2;
      for (char ch : st.chat_dir("c8")) esc2 += ch == '\\' ? std::string("\\\\") : std::string(1, ch);
      check("terminal paths follow a renamed folder", files::Store::read((fs::path(st.chat_dir("c8")) / ".terminal.json").string(), t8) &&
            t8.find(esc2 + "\\\\.venv") != std::string::npos && t8.find(esc + "\"") == std::string::npos, t8);
    }
    check("an untitled folder named at a later save", fs::path(st.chat_dir("c9")).filename().string() == fs::path(d9).filename().string().substr(0, 10) + " Judul kemudian (c9)", st.chat_dir("c9"));
    files::Info g;
    check("the file found after the rename", st.get(a.id, g) && g.path == (fs::path(d1) / "notes.txt").string(), g.path);
    std::string t;
    check("chat.json read back", st.chat_get("c1", t) && t.find("Rapat") != std::string::npos);
    // a later save with another title keeps the folder
    st.chat_put("c1", "{\"title\":\"Other\",\"history\":[{\"content\":\"/files/" + a.id + "\"}]}");
    check("the folder keeps its name", st.chat_dir("c1") == d1, st.chat_dir("c1"));
    // the same name with other bytes: a second name
    const files::Info b = st.put("notes.txt", "hello again", "c1");
    check("a name taken: notes (2).txt", fs::path(b.path).filename() == "notes (2).txt", b.path);
    // the same bytes again in the same chat: the same file
    check("the same bytes: the same file", st.put("x.txt", "hello", "c1").path == (fs::path(d1) / "notes.txt").string());
    // another chat refers to the file (@): a copy in its folder (not a hard link: a command run in one chat that
    // rewrote the file changed the other chat's too)
    st.chat_put("c2", "{\"title\":\"Second\",\"history\":[{\"content\":[{\"type\":\"file\",\"file\":{\"file_id\":\"" + a.id + "\"}}]}]}");
    const fs::path l2 = fs::path(st.chat_dir("c2")) / "notes.txt";
    check("copied into the other chat", fs::exists(l2, ec) && fs::hard_link_count(l2, ec) == 1, l2.string());
    check("two holders", st.holders(a.id).size() == 2);
    // a document's source goes along with the link
    const files::Info doc = st.put("report.docx", "PK-fake", "c1");
    st.put_source(doc.id, "c1", "{\"content\":\"# R\"}");
    st.chat_put("c3", "{\"title\":\"Third\",\"history\":[{\"content\":\"/files/" + doc.id + "\"}]}");
    std::string src;
    check("the source found", st.source(doc.id, src) && src.find("# R") != std::string::npos);
    check("the source copied to the other chat", fs::exists(fs::path(st.chat_dir("c3")) / ".src" / (doc.id + ".json"), ec));
    // a file put into a folder by hand: listed with an id; one deleted by hand: gone from the list
    { std::ofstream o(fs::path(d1) / "dropped.csv", std::ios::binary); o << "a,b\n1,2\n"; }
    fs::remove(fs::path(d1) / "notes (2).txt", ec);
    bool dropped = false, gone = true;
    for (const files::Info& i : st.list()) {
      dropped = dropped || (i.name == "dropped.csv" && i.chat == "c1");
      gone = gone && i.id != b.id;
    }
    check("a file put in by hand is listed", dropped);
    check("a file deleted by hand is not", gone);
    // deleting a chat deletes its folder; the copy in the other chat stays
    check("chat deleted", st.chat_remove("c1"));
    check("its folder is gone", !fs::exists(d1, ec));
    check("the other chat's copy stays", st.get(a.id, g) && g.chat == "c2", g.path);
    check("the other copy is whole", files::Store::read(g.path, t) && t == "hello");
    // a file out of one chat only, then everywhere
    st.put("keep.txt", "k", "c3");
    const files::Info k2 = st.put("keep.txt", "k", "c2");
    check("out of one chat", st.remove_from(k2.id, "c2") && st.holders(k2.id).size() == 1);
    check("removed everywhere", st.remove(k2.id) && !st.get(k2.id, g));
    // keep_files: chat.json goes, the files stay
    check("chat deleted, files kept", st.chat_remove("c2", true) && st.get(a.id, g) && !st.chat_get("c2", t));
    // a bad chat id never names a path
    bool threw = false;
    try { st.chat_put("..\\x", "{}"); } catch (const std::exception&) { threw = true; }
    check("a bad chat id is refused", threw);
  }
  // a document in a message: whole when it fits a quarter of the model's context, else its outline (read_file then
  // reads it a part at a time: files::read_budget)
  {
    std::string md;
    for (int k = 1; k <= 8; ++k) {
      md += "# Bab " + std::to_string(k) + "\n\n";
      for (int s = 0; s < 20; ++s) md +="Kalimat isi bab yang cukup panjang untuk mengisi dokumen uji. ";
      md += "\n\n";
    }
    const files::Info doc = files::store().put("panjang.md", md, "c7");
    Json msgs = Json::array(), m = Json::object(), parts = Json::array(), p = Json::object(), f = Json::object();
    f.set("file_id", doc.id);
    p.set("type", "file");
    p.set("file", f);
    parts.push(p);
    m.set("role", "user");
    m.set("content", parts);
    msgs.push(m);
    auto text_of = [](const Json& out) { return out[0]["content"][0]["text"].str(); };
    check("document: whole without a known context", text_of(files::expand_files(msgs, true)).find("Outline:") == std::string::npos);
    check("document: its outline for an 8K context", text_of(files::expand_files(msgs, true, 8192)).find("Outline:") != std::string::npos);
    check("document: whole without the file tools", text_of(files::expand_files(msgs, false, 8192)).find("Outline:") == std::string::npos);
    check("read budget: a fifth of the context", files::read_budget(8192) == 4915 && files::read_budget(1000) == 1500 &&
          files::read_budget(0) == (size_t)-1, std::to_string(files::read_budget(8192)));
  }
  {
    const std::string cwd = root.string();
    term::Result r = term::run("Get-Location | Select-Object -ExpandProperty Path; 'h\xc3\xa9llo'; Write-Error 'oops'; exit 3", cwd, "C:\\nowhere", 60);
    check("exit code", r.exit_code == 3, std::to_string(r.exit_code));
    // (PowerShell prints the long path; TEMP can be an 8.3 short one, C:\Users\RUNNER~1\... on GitHub's runner)
    const std::string long_cwd = fs::canonical(root).string();
    check("runs in its folder", r.output.find(long_cwd) != std::string::npos, r.output);
    check("UTF-8 output", r.output.find("h\xc3\xa9llo") != std::string::npos, r.output);
    check("stderr too", r.output.find("oops") != std::string::npos, r.output);
    r = term::run("$env:PATH.Split(';')[0]", cwd, "C:\\nowhere", 60);
    check("PATH first", r.output.find("C:\\nowhere") == 0, r.output);
    r = term::run("Start-Process ping.exe -ArgumentList '-n','60','127.0.0.1' -NoNewWindow; Start-Sleep 60", cwd, "", 3);
    check("timeout ends it and its children", r.timed_out && r.seconds < 15, std::to_string(r.seconds));
    {  // (the client gone: the command ends at once, not at its timeout)
      const auto t0 = std::chrono::steady_clock::now();
      int asked = 0;
      r = term::run("Start-Sleep 30", cwd, "", 60, "", [&] { return ++asked > 3; });
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      check("a cancelled command ends at once", r.cancelled && !r.timed_out && s < 10, std::to_string(s));
    }
    r = term::run("1..100000", cwd, "", 60);
    check("long output cut", r.truncated && r.output.size() < term::kMaxOut + 200 && r.output.find("100000") != std::string::npos,
          std::to_string(r.output.size()));
    // the chat's terminal session: the folder and the environment carry over to the next command
    const std::string state = (root / ".terminal.json").string();
    r = term::run("New-Item -ItemType Directory sub | Out-Null; Set-Location sub; $env:AZ_X = 'satu'", cwd, "C:\\nowhere", 60, state);
    r = term::run("(Get-Location).Path; $env:AZ_X", cwd, "C:\\nowhere", 60, state);
    check("session: folder and variable kept", r.output.find(long_cwd + "\\sub") != std::string::npos && r.output.find("satu") != std::string::npos, r.output);
    r = term::run("Set-Location ..; exit 3", cwd, "C:\\nowhere", 60, state);
    check("session: exit code", r.exit_code == 3, std::to_string(r.exit_code));
    r = term::run("Remove-Item Env:AZ_X; Set-Location sub; Write-Error 'oops'", cwd, "C:\\nowhere", 60, state);
    const int plain = term::run("Write-Error 'oops'", cwd, "", 60).exit_code;
    check("session: a failed command's exit code as without it", r.exit_code == plain, std::to_string(r.exit_code) + " vs " + std::to_string(plain));
    r = term::run("(Get-Location).Path; \"x=$env:AZ_X.\"; ($env:PATH.Split(';') | Where-Object { $_ -eq 'C:\\nowhere' }).Count", cwd, "C:\\nowhere", 60, state);
    check("session: kept after exit, variable removed, PATH not doubled", r.output.find(long_cwd + "\\sub") != std::string::npos &&
          r.output.find("x=.") != std::string::npos && r.output.find("\n1") != std::string::npos, r.output);
    // a venv: Activate.ps1 as Python's (VIRTUAL_ENV, PATH, _OLD_VIRTUAL_PATH), then deactivate in a later command
    fs::create_directories(root / "sub" / ".venv" / "Scripts");
    std::ofstream(root / "sub" / ".venv" / "Scripts" / "Activate.ps1", std::ios::binary) <<
                        ("$env:_OLD_VIRTUAL_PATH = $env:PATH; $env:VIRTUAL_ENV = (Resolve-Path \"$PSScriptRoot\\..\").Path; "
                        "$env:PATH = \"$env:VIRTUAL_ENV\\Scripts;$env:PATH\"; function global:deactivate { 'old' }");
    r = term::run(".\\.venv\\Scripts\\Activate.ps1", cwd, "C:\\nowhere", 60, state);
    r = term::run("$env:VIRTUAL_ENV; $env:PATH.Split(';')[0]", cwd, "C:\\nowhere", 60, state);
    check("session: venv active in the next command", r.output.find(".venv\r\n") != std::string::npos && r.output.find(".venv\\Scripts") != std::string::npos, r.output);
    r = term::run("deactivate", cwd, "C:\\nowhere", 60, state);
    r = term::run("\"v=$env:VIRTUAL_ENV.\"; $env:PATH.Split(';')[0]", cwd, "C:\\nowhere", 60, state);
    check("session: deactivate in a later command", r.output.find("v=.") != std::string::npos && r.output.find("C:\\nowhere") != std::string::npos, r.output);
  }
  fs::remove_all(root, ec);
  printf(fails ? "test_chat_store: %d FAILED\n" : "test_chat_store: all passed\n", fails);
  return fails ? 1 : 0;
}
