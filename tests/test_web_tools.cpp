// CPU-only, offline checks of tools/web_tools.h: HTML to text, search result parsing (Brave and Bing page shapes),
// the address filter and redirect resolution; the links kept in a page's text, the web ports, the one URL parser,
// internationalized hosts, and web_view's command-line URL (tools/web_view.h). With --online it also runs one live
// search and one fetch.
#include "../tools/web_tools.h"
#include "../tools/web_view.h"
#include <cstdio>
#include <string>

static int fails = 0;
static void check(const char* name, bool ok, const std::string& got = "") {
  printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name, ok || got.empty() ? "" : "  got: ", ok ? "" : got.c_str());
  fails += !ok;
}

static bool addr_public(const char* ip) {
  sockaddr_storage ss{};
  if (strchr(ip, ':')) {
    auto* a = (sockaddr_in6*)&ss;
    a->sin6_family = AF_INET6;
    inet_pton(AF_INET6, ip, &a->sin6_addr);
  } else {
    auto* a = (sockaddr_in*)&ss;
    a->sin_family = AF_INET;
    inet_pton(AF_INET, ip, &a->sin_addr);
  }
  return webt::public_addr((const sockaddr*)&ss);
}

int main(int argc, char** argv) {
  using namespace webt;
  {
    std::string title;
    const std::string t = html_to_text(
        "<html><head><title>A &amp; B</title><style>p{x}</style></head><body><nav>menu</nav>"
        "<h1>Head</h1><p>One &lt;two&gt;&nbsp;three&#33; &#x263A;</p><script>var a='<p>';</script>"
        "<ul><li>x</li><li>y</li></ul><!-- c --><footer>foot</footer></body></html>", &title);
    check("title", title == "A & B", title);
    check("text", t == "# Head\n\nOne <two> three! \xe2\x98\xba\n\n- x\n- y", t);
  }
  {
    std::string body(500, 'w');
    const std::string t = html_to_text("<body><div>side</div><main><p>" + body + "</p></main></body>");
    check("main only", t == body, t.substr(0, 40));
  }
  {
    const std::string brave =
        "<div data-type=\"web\" x><a href=\"https://a.example/x?a=1&amp;b=2\" class=\"l1\"><div class=\"title t\" title=\"T\">"
        "Title <b>A</b></div></a><div class=\"generic-snippet\"><div class=\"content c\">Snip &amp; more</div></div></div>"
        "<div data-type=\"web\"><a href=\"https://b.example/\"><div class=\"title\">B</div></a></div>";
    const auto r = parse_brave(brave, 10);
    check("brave count", r.size() == 2, std::to_string(r.size()));
    check("brave first", r.size() == 2 && r[0].url == "https://a.example/x?a=1&b=2" && r[0].title == "Title A" && r[0].snippet == "Snip & more",
          r.empty() ? "" : r[0].url + " | " + r[0].title + " | " + r[0].snippet);
  }
  {
    // u=a1 + base64url("https://www.intel.com/x")
    const std::string bing =
        "<li class=\"b_algo\"><h2><a href=\"https://www.bing.com/ck/a?!&amp;&amp;p=1&amp;u=a1aHR0cHM6Ly93d3cuaW50ZWwuY29tL3g&amp;ntb=1\">"
        "Intel <strong>X</strong></a></h2><div class=\"b_caption\"><p class=\"b_lineclamp2\">The snippet.</p></div></li>"
        "<li class=\"b_algo\"><h2><a href=\"https://direct.example/\">D</a></h2></li>";
    const auto r = parse_bing(bing, 10);
    check("bing count", r.size() == 2, std::to_string(r.size()));
    check("bing redirect decoded", r.size() == 2 && r[0].url == "https://www.intel.com/x" && r[0].title == "Intel X" && r[0].snippet == "The snippet.",
          r.empty() ? "" : r[0].url + " | " + r[0].title + " | " + r[0].snippet);
    check("bing direct", r.size() == 2 && r[1].url == "https://direct.example/");
  }
  check("public 8.8.8.8", addr_public("8.8.8.8"));
  check("public 2606:4700::1111", addr_public("2606:4700::1111"));
  for (const char* ip : {"127.0.0.1", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.1.1", "169.254.1.1", "100.64.0.1",
                         "0.0.0.0", "224.0.0.1", "::1", "::", "fe80::1", "fd00::1", "::ffff:192.168.1.1", "ff02::1",
                         "240.0.0.1", "192.0.0.8", "192.0.2.1", "198.18.0.1", "198.19.255.1", "198.51.100.7", "203.0.113.9",
                         "::127.0.0.1", "::10.0.0.1", "64:ff9b::7f00:1", "64:ff9b::c0a8:101", "2002:7f00:1::", "2002:a00:1::1",
                         "2001::1", "2001:db8::1", "100::1", "fec0::1", "feff::1"})
    check((std::string("private ") + ip).c_str(), !addr_public(ip));
  check("public 172.32.0.1", addr_public("172.32.0.1"));
  for (const char* ip : {"198.20.0.1", "192.0.3.1", "64:ff9b::808:808", "2002:808:808::1", "2001:4860:4860::8888", "::8.8.8.8"})
    check((std::string("public ") + ip).c_str(), addr_public(ip));
  check("localhost refused", !check_host("localhost").empty());
  check("127.0.0.1 refused", !check_host("127.0.0.1").empty());
  check("resolve absolute", resolve_url("https://a.com/x/y", "http://b.com/z") == "http://b.com/z");
  check("resolve scheme-relative", resolve_url("https://a.com/x/y", "//b.com/z") == "https://b.com/z");
  check("resolve path", resolve_url("https://a.com/x/y?q", "/z") == "https://a.com/z");
  check("resolve relative", resolve_url("https://a.com/x/y?q=/w", "z") == "https://a.com/x/z", resolve_url("https://a.com/x/y?q=/w", "z"));
  check("local fetch refused", !get("http://127.0.0.1:8080/").error.empty());
  check("non-http refused", !get("file:///C:/Windows/win.ini").error.empty());
  check("service port refused before any lookup", get("http://example.com:25/").error.find("port 25") != std::string::npos,
        get("http://example.com:25/").error);

  // links in a page's text: files and references keep their URLs, other pages their text (counted), all with links
  {
    const std::string page =
        "<html><body><main><p>Read <a href=\"/docs/guide.pdf\">the guide</a>, see <a href=\"https://other.example/about\">about us</a>"
        " and <a href=\"https://github.com/o/r/issues/12\">the bug</a>; also <a href=\"/pull/7\">#7</a>, "
        "<a href=\"#top\">top</a>, <a href=\"javascript:x()\">js</a>, <a href=\"https://x.example/full\">https://x.example/full</a>."
        "<img src=\"/shot.png\" alt=\"a screenshot\"><img src=\"/logo\" alt=\"logo\"> " + std::string(400, 'w') + "</p></main></body></html>";
    int hidden = -1;
    const std::string t = html_to_text(page, nullptr, "https://site.example/a/b", false, &hidden);
    check("file link kept", t.find("[the guide](https://site.example/docs/guide.pdf)") != std::string::npos, t.substr(0, 300));
    check("page link as text", t.find("about us") != std::string::npos && t.find("other.example") == std::string::npos, t.substr(0, 300));
    check("issue link kept", t.find("[the bug](https://github.com/o/r/issues/12)") != std::string::npos, t.substr(0, 300));
    check("#7 reference kept", t.find("[#7](https://site.example/pull/7)") != std::string::npos, t.substr(0, 300));
    check("URL written out kept", t.find("https://x.example/full") != std::string::npos, t.substr(0, 300));
    check("fragment / javascript: as text", t.find("javascript") == std::string::npos && t.find("#top") == std::string::npos, t.substr(0, 300));
    check("image of a file kept", t.find("[image: a screenshot](https://site.example/shot.png)") != std::string::npos, t.substr(0, 400));
    check("image of a page hidden", t.find("logo") == std::string::npos, t.substr(0, 400));
    check("hidden counted", hidden == 1, std::to_string(hidden));  // (about us only: top and js lead nowhere)
    int h2 = -1;
    const std::string all = html_to_text(page, nullptr, "https://site.example/a/b", true, &h2);
    check("links: true keeps every URL", all.find("[about us](https://other.example/about)") != std::string::npos && h2 == 0, all.substr(0, 300));
  }
  check("file_link pdf", file_link("https://a.example/x/report.PDF?dl=1"));
  check("file_link attachment", file_link("https://github.com/user-attachments/files/1/log.txt"));
  check("file_link audio", file_link("https://a.example/podcast.flac"));
  check("file_link not a page", !file_link("https://a.example/blog/post"));
  check("ref_link issue", ref_link("https://github.com/o/r/pull/3", "fix"));
  check("ref_link #123", ref_link("https://a.example/x", " #123 "));
  check("ref_link not text", !ref_link("https://a.example/x", "click here"));

  // ports: the web's, not other services' (Chrome's restricted list)
  for (int p : {80, 443, 8080, 8443, 3000, 65535}) check(("web port " + std::to_string(p)).c_str(), web_port_ok(p));
  for (int p : {0, 22, 25, 53, 110, 139, 445, 6667, 10080, 65536, -1}) check(("service port " + std::to_string(p)).c_str(), !web_port_ok(p));

  // the one URL parser (as get() uses), and internationalized hosts
  {
    std::string h;
    int p = 0;
    check("parse https", url_host_port("https://Example.com/x?y", h, p) && p == 443, h + ":" + std::to_string(p));
    check("parse port", url_host_port("http://a.example:8080/", h, p) && h == "a.example" && p == 8080, h + ":" + std::to_string(p));
    check("parse user@host", url_host_port("http://u:pw@b.example/", h, p) && h == "b.example", h);
    check("parse not http", !url_host_port("ftp://a.example/", h, p));
    check("ascii host kept", ascii_host("example.com") == "example.com");
    check("IDN host to punycode", ascii_host("m\xc3\xbcnchen.de") == "xn--mnchen-3ya.de", ascii_host("m\xc3\xbcnchen.de"));
  }

  // web_view's URL on Edge's command line: nothing leaves its quotes
  {
    const std::string a = webv::arg_url("https://a.example/p\"q --dump-dom r\x01s");
    check("quote, space and control encoded", a == "https://a.example/p%22q%20--dump-dom%20r%01s", a);
    check("IDN host as punycode", webv::arg_url("https://m\xc3\xbcnchen.de/\xc3\xa4") == "https://xn--mnchen-3ya.de/%C3%A4",
          webv::arg_url("https://m\xc3\xbcnchen.de/\xc3\xa4"));
    check("odd byte in the host refused", webv::arg_url("https://a\".example/").empty(), webv::arg_url("https://a\".example/"));
    check("port and user kept", webv::arg_url("https://u@a.example:8443/x") == "https://u@a.example:8443/x", webv::arg_url("https://u@a.example:8443/x"));
    check("not http refused", webv::arg_url("file:///C:/x").empty());
  }

  if (argc > 1 && std::string(argv[1]) == "--online") {
    std::string used, err;
    const auto r = search("intel arc level zero", 5, "auto", &used, &err);
    printf("search (%s): %zu results [%s]\n", used.c_str(), r.size(), err.c_str());
    for (auto& x : r) printf("  %s\n    %s\n    %.100s\n", x.title.c_str(), x.url.c_str(), x.snippet.c_str());
    const auto b = search("intel arc level zero", 5, "bing", &used, &err);
    printf("bing: %zu results %s\n", b.size(), err.c_str());
    const Page p = get("https://en.wikipedia.org/wiki/Intel_Arc");
    std::string title;
    const std::string t = html_to_text(p.body, &title);
    printf("fetch: HTTP %d, %zu bytes -> %zu chars, title '%s'\n%.600s\n", p.status, p.body.size(), t.size(), title.c_str(), t.c_str());
    fails += r.empty() || b.empty() || t.size() < 1000;
  }
  printf(fails ? "test_web_tools: %d failed\n" : "test_web_tools: all passed\n", fails);
  return fails ? 1 : 0;
}
