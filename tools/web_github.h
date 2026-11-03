// A GitHub issue or pull request for web_fetch (tools/server.cpp), read the way a person reads it: the thread and
// every comment, from GitHub's REST API instead of its page. The page is built by scripts, so its HTML often holds no
// comments at all; the API gives each one as Markdown with its author and time, and the links in it whole (files
// attached to the issue, screenshots), which web_fetch can then open. No token: GitHub allows 60 requests an hour
// from one address that way, and a thread takes 2-4 of them.
#pragma once
#include <set>
#include "../src/json.h"
#include "web_tools.h"

namespace webt {

// github.com/<owner>/<repo>/issues/<n> or /pull/<n> (anything after <n> ignored): its parts; false: not such a URL
inline bool github_thread_url(const std::string& url, std::string& owner, std::string& repo, std::string& num, bool& pull) {
  std::string u = url;
  for (const char* s : {"https://", "http://"})
    if (lower(u.substr(0, strlen(s))) == s) u = u.substr(strlen(s));
  const std::string host = lower(u.substr(0, u.find('/')));
  if (host != "github.com" && host != "www.github.com") return false;
  std::vector<std::string> seg;
  size_t p = u.find('/');
  while (p != std::string::npos && seg.size() < 4) {
    const size_t q = u.find_first_of("/?#", p + 1);
    seg.push_back(u.substr(p + 1, (q == std::string::npos ? u.size() : q) - p - 1));
    p = q != std::string::npos && u[q] == '/' ? q : std::string::npos;
  }
  if (seg.size() < 4 || (seg[2] != "issues" && seg[2] != "pull") || seg[3].empty() || seg[3].size() > 9) return false;
  for (char c : seg[3]) if (!isdigit((unsigned char)c)) return false;
  auto name_ok = [](const std::string& s) {
    if (s.empty() || s.size() > 100) return false;
    for (char c : s) if (!isalnum((unsigned char)c) && c != '-' && c != '_' && c != '.') return false;
    return true;
  };
  if (!name_ok(seg[0]) || !name_ok(seg[1])) return false;
  owner = seg[0], repo = seg[1], num = seg[3], pull = seg[2] == "pull";
  return true;
}

// "2026-11-02T10:41:28Z" -> "2026-11-02 10:41 UTC"
inline std::string github_time(const std::string& t) {
  return t.size() >= 16 && t[10] == 'T' ? t.substr(0, 10) + " " + t.substr(11, 5) + " UTC" : t;
}

// one API call: its JSON, or err (the status in words)
inline bool github_api(const std::string& path, Json& out, std::string& err) {
  const Page p = get("https://api.github.com" + path, 8u << 20, 15000, "en-US,en;q=0.9", "application/vnd.github+json");
  if (!p.error.empty()) { err = "GitHub API: " + p.error; return false; }
  if (p.status == 403 || p.status == 429)
    { err = "GitHub API: limit reached (60 requests an hour from this address without a token); try again later"; return false; }
  if (p.status != 200) { err = "GitHub API: HTTP " + std::to_string(p.status) + " for " + path; return false; }
  if (!Json::try_parse(p.body, out)) { err = "GitHub API: not JSON"; return false; }
  return true;
}

// The files attached in one post (GitHub's user-attachments: files/<n>/<name>, assets/<uuid> for images), as
// "- <name> (<where>): <url>" lines, each URL once over the thread (seen)
inline void github_attachments(const std::string& body, const std::string& where, std::set<std::string>& seen, std::string& out) {
  static const std::string mark = "https://github.com/user-attachments/";
  for (size_t k = body.find(mark); k != std::string::npos; k = body.find(mark, k + mark.size())) {
    size_t e = k;
    while (e < body.size() && !strchr(" \t\r\n)\"'<>]", body[e])) ++e;
    const std::string url = body.substr(k, e - k);
    if (!seen.insert(url).second) continue;
    const std::string rest = url.substr(mark.size());
    const std::string name = rest.rfind("files/", 0) == 0 ? rest.substr(rest.rfind('/') + 1) : "image";
    out += "- " + name + " (" + where + "): " + url + "\n";
  }
}

// The issue or pull request at url as Markdown text (title: its title). false with err "" : not a GitHub thread URL
// (fetch the page instead); false with err: the API failed
inline bool github_thread(const std::string& url, std::string& title, std::string& text, std::string& err) {
  std::string owner, repo, num;
  bool pull = false;
  err.clear();
  if (!github_thread_url(url, owner, repo, num, pull)) return false;
  const std::string base = "/repos/" + owner + "/" + repo;
  Json is;
  if (!github_api(base + "/issues/" + num, is, err)) return false;
  pull = pull || is["pull_request"].is_object();
  title = is["title"].str();
  std::string o = "# " + title + "\n" + is["html_url"].str(url) + " | " + (pull ? "pull request" : "issue") + " #" + num +
                  " in " + owner + "/" + repo + " | " + is["state"].str() + " | opened by " + is["user"]["login"].str("?") +
                  " on " + github_time(is["created_at"].str()) + " | " + std::to_string(is["comments"].as_int(0)) + " comments\n";
  std::string labels;
  for (const Json& l : is["labels"].items()) labels += (labels.empty() ? "" : ", ") + l["name"].str();
  if (!labels.empty()) o += "labels: " + labels + "\n";
  const std::string body = is["body"].str();
  std::set<std::string> seen;
  std::string files;  // (every file attached in the thread, listed before it: a small context's first part shows them)
  github_attachments(body, "the description", seen, files);
  o += "\n" + (body.empty() ? std::string("(no description)") : body) + "\n";
  // the comments, oldest first, 100 a page (at most 10 pages)
  int n = 0;
  for (int page = 1; page <= 10; ++page) {
    Json cs;
    if (!github_api(base + "/issues/" + num + "/comments?per_page=100&page=" + std::to_string(page), cs, err)) {
      o += "\n---\n[the comments after these could not be read: " + err + "]\n";
      err.clear();
      break;
    }
    for (const Json& c : cs.items()) {
      const std::string who = c["user"]["login"].str("?"), at = github_time(c["created_at"].str());
      o += "\n---\n### Comment " + std::to_string(++n) + " by " + who + " on " + at + "\n" + c["body"].str() + "\n";
      github_attachments(c["body"].str(), "comment " + std::to_string(n) + " by " + who + ", " + at, seen, files);
    }
    if (cs.items().size() < 100) break;
  }
  // a pull request's review comments, on lines of its diff (one page: the first 100)
  if (pull) {
    Json rc;
    std::string e2;
    if (github_api(base + "/pulls/" + num + "/comments?per_page=100", rc, e2) && !rc.items().empty()) {
      o += "\n---\n## Review comments on the diff\n";
      for (const Json& c : rc.items())
        o += "\n### " + c["user"]["login"].str("?") + " on " + c["path"].str() +
             (c["line"].as_int(0) > 0 ? " line " + std::to_string(c["line"].as_int(0)) : std::string()) + ", " +
             github_time(c["created_at"].str()) + "\n" + c["body"].str() + "\n";
    }
  }
  // (the list after the header line(s), before the description)
  if (!files.empty()) {
    const size_t at = o.find("\n\n");
    const std::string list = "\nFiles attached in this thread (open one with web_fetch and its URL):\n" + files;
    o = at == std::string::npos ? o + list : o.substr(0, at + 1) + list + o.substr(at + 1);
  }
  text = o;
  return true;
}

}  // namespace webt
