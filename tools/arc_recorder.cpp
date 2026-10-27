// arc-recorder: records a meeting for the Meetings tab, apart from the server: the microphone ("Me") and the meeting's
// sound ("Others": one app's sound through WASAPI process loopback (Windows 10 2004 and later), or everything the
// speakers play) as a stereo 16-bit WAV at 48 kHz in <data>\recordings\, written as it comes, its sizes kept right
// every second (a crash or a power cut loses a second at most). Each packet is placed by its time stamp, so the tracks
// stay together and a silent app, a headset change or a device that comes back leave silence, not a shift. The
// server is not needed while it records; when it stops, the recording goes to the server's transcript library
// (POST /v1/transcripts/import: the accurate pass, Me / Others, then MP3), or the server takes it at its next start.
//   arc-recorder start [--app <name.exe | pid>] [--no-mic] [--title "..."] [--seconds N]
//   arc-recorder stop       ends the recording that runs (from another console; Ctrl+C in its own does the same)
//   arc-recorder status
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <audioclient.h>
#include <audioclientactivationparams.h>
#include <mmdeviceapi.h>
#include <tlhelp32.h>
#include "../src/app_dir.h"
#include "../src/audio.h"
#include "../src/json.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "mmdevapi.lib")
#pragma comment(lib, "ws2_32.lib")

namespace fs = std::filesystem;

namespace {

constexpr int kRate = 48000;
const wchar_t* kStopEvent = L"Local\\arcinferzero-recorder-stop";

fs::path state_file() { return fs::path(app_dir()) / "recorder.json"; }

double qpc_seconds(int64_t q) {
  static const double f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return (double)x.QuadPart; }();
  return (double)q / f;
}
double now_seconds() {
  LARGE_INTEGER x;
  QueryPerformanceCounter(&x);
  return qpc_seconds(x.QuadPart);
}

// IActivateAudioInterfaceCompletionHandler for ActivateAudioInterfaceAsync (process loopback)
class Activated : public IActivateAudioInterfaceCompletionHandler, public IAgileObject {
 public:
  HANDLE done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  HRESULT hr = E_FAIL;
  IAudioClient* client = nullptr;
  ~Activated() { CloseHandle(done); }
  STDMETHODIMP ActivateCompleted(IActivateAudioInterfaceAsyncOperation* op) override {
    IUnknown* u = nullptr;
    HRESULT r = E_FAIL;
    if (SUCCEEDED(op->GetActivateResult(&r, &u)) && SUCCEEDED(r) && u) {
      hr = u->QueryInterface(__uuidof(IAudioClient), (void**)&client);
      u->Release();
    } else {
      hr = FAILED(r) ? r : E_FAIL;
    }
    SetEvent(done);
    return S_OK;
  }
  STDMETHODIMP QueryInterface(REFIID id, void** p) override {
    if (id == __uuidof(IUnknown) || id == __uuidof(IAgileObject) || id == __uuidof(IActivateAudioInterfaceCompletionHandler)) {
      *p = static_cast<IActivateAudioInterfaceCompletionHandler*>(this);
      AddRef();
      return S_OK;
    }
    *p = nullptr;
    return E_NOINTERFACE;
  }
  STDMETHODIMP_(ULONG) AddRef() override { return (ULONG)InterlockedIncrement(&ref_); }
  STDMETHODIMP_(ULONG) Release() override {
    const ULONG n = (ULONG)InterlockedDecrement(&ref_);
    if (!n) delete this;
    return n;
  }

 private:
  LONG ref_ = 1;
};

// one track: a capture device (the microphone; the speakers' loopback) or one process's loopback. Its packets,
// mixed to mono and resampled to 48 kHz, go into a ring at their place in time (the recording's sample index)
struct Track {
  std::string what;           // ("the microphone", "Zoom.exe's sound", ...)
  enum Kind { Mic, Speakers, Process } kind = Mic;
  DWORD pid = 0;
  IAudioClient* ac = nullptr;
  IAudioCaptureClient* cc = nullptr;
  HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  int rate = 0, ch = 0, bits = 0;
  bool fl = false;  // (float samples)
  std::unique_ptr<audio::Resampler> rs;
  int64_t next = -1;          // (the ring index its next resampled sample goes to)
  std::vector<float> ring = std::vector<float>((size_t)kRate * 20, 0.0f);
  bool down = false;          // (its device or process is gone: silence until it is back)
  double retry_at = 0;
  int64_t packets = 0;

  void close() {
    if (ac) ac->Stop();
    if (cc) { cc->Release(); cc = nullptr; }
    if (ac) { ac->Release(); ac = nullptr; }
  }
  // the client opened, with this format (endpoint devices: the device's own; process loopback: 16-bit stereo 48 kHz)
  bool start(IAudioClient* c, const WAVEFORMATEX* wf, DWORD flags) {
    ac = c;
    rate = (int)wf->nSamplesPerSec;
    ch = wf->nChannels;
    bits = wf->wBitsPerSample;
    fl = wf->wFormatTag == WAVE_FORMAT_IEEE_FLOAT ||
         (wf->wFormatTag == WAVE_FORMAT_EXTENSIBLE && ((const WAVEFORMATEXTENSIBLE*)wf)->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT);
    if (FAILED(ac->Initialize(AUDCLNT_SHAREMODE_SHARED, flags | AUDCLNT_STREAMFLAGS_EVENTCALLBACK, 2000000, 0, wf, nullptr)) ||
        FAILED(ac->SetEventHandle(ev)) || FAILED(ac->GetService(__uuidof(IAudioCaptureClient), (void**)&cc)) ||
        FAILED(ac->Start())) {
      close();
      return false;
    }
    rs = std::make_unique<audio::Resampler>(rate, kRate);
    next = -1;
    down = false;
    return true;
  }
  bool open() {
    close();
    if (kind == Process) {
      AUDIOCLIENT_ACTIVATION_PARAMS ap = {};
      ap.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
      ap.ProcessLoopbackParams.TargetProcessId = pid;
      ap.ProcessLoopbackParams.ProcessLoopbackMode = PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE;
      PROPVARIANT pv = {};
      pv.vt = VT_BLOB;
      pv.blob.cbSize = sizeof ap;
      pv.blob.pBlobData = (BYTE*)&ap;
      Activated* h = new Activated();
      IActivateAudioInterfaceAsyncOperation* op = nullptr;
      HRESULT hr = ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, __uuidof(IAudioClient), &pv, h, &op);
      if (SUCCEEDED(hr)) WaitForSingleObject(h->done, 5000);
      IAudioClient* c = SUCCEEDED(hr) && SUCCEEDED(h->hr) ? h->client : nullptr;
      if (op) op->Release();
      h->client = nullptr;
      h->Release();
      if (!c) return false;
      WAVEFORMATEX wf = {};
      wf.wFormatTag = WAVE_FORMAT_PCM;
      wf.nChannels = 2;
      wf.nSamplesPerSec = kRate;
      wf.wBitsPerSample = 16;
      wf.nBlockAlign = 4;
      wf.nAvgBytesPerSec = kRate * 4;
      return start(c, &wf, AUDCLNT_STREAMFLAGS_LOOPBACK);
    }
    IMMDeviceEnumerator* en = nullptr;
    IMMDevice* dev = nullptr;
    IAudioClient* c = nullptr;
    WAVEFORMATEX* wf = nullptr;
    bool ok = SUCCEEDED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&en)) &&
              SUCCEEDED(en->GetDefaultAudioEndpoint(kind == Mic ? eCapture : eRender, kind == Mic ? eCommunications : eConsole, &dev)) &&
              SUCCEEDED(dev->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&c)) && SUCCEEDED(c->GetMixFormat(&wf));
    if (ok) ok = start(c, wf, kind == Speakers ? AUDCLNT_STREAMFLAGS_LOOPBACK : 0);
    else if (c) c->Release();
    if (wf) CoTaskMemFree(wf);
    if (dev) dev->Release();
    if (en) en->Release();
    return ok;
  }
  // the packets waiting: mixed to mono, resampled, placed by their time (t0: the recording's start, QPC seconds)
  void drain(double t0) {
    if (!cc || down) return;
    for (;;) {
      UINT32 n = 0;
      HRESULT hr = cc->GetNextPacketSize(&n);
      if (FAILED(hr)) { fail(hr); return; }
      if (!n) return;
      BYTE* d = nullptr;
      DWORD flags = 0;
      UINT64 pos = 0, qpc = 0;
      hr = cc->GetBuffer(&d, &n, &flags, &pos, &qpc);
      if (FAILED(hr)) { fail(hr); return; }
      std::vector<float> mono(n, 0.0f);
      if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT))
        for (UINT32 i = 0; i < n; ++i) {
          float a = 0;
          for (int c = 0; c < ch; ++c) {
            const BYTE* s = d + ((size_t)i * ch + c) * (bits / 8);
            float v;
            if (fl) memcpy(&v, s, 4);
            else if (bits == 16) v = (int16_t)(s[0] | s[1] << 8) / 32768.0f;
            else if (bits == 24) v = (float)((int32_t)((uint32_t)s[0] << 8 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 24) >> 8) / 8388608.0f;
            else { int32_t x; memcpy(&x, s, 4); v = x / 2147483648.0f; }
            a += v;
          }
          mono[i] = a / ch;
        }
      cc->ReleaseBuffer(n);
      ++packets;
      // (the packet's place: its first frame's time; QPC positions are in 100-ns units)
      const double ts = qpc ? (double)qpc / 1e7 : now_seconds() - (double)n / rate;  // (no time stamp: now, less its length)
      const int64_t at = (int64_t)std::llround((ts - t0) * kRate);
      if (next < 0 || std::llabs(at - next) > kRate / 10) {  // (the first, or after a gap of more than 0.1 s: a new start)
        rs = std::make_unique<audio::Resampler>(rate, kRate);
        next = std::max<int64_t>(0, at);
      }
      std::vector<float> out;
      rs->push(mono.data(), mono.size(), out);
      for (float v : out) ring[(size_t)(next++ % (int64_t)ring.size())] = v;
    }
  }
  void fail(HRESULT hr) {  // (a device unplugged or changed, an app closed: closed, tried again every second)
    if (!down) fprintf(stderr, "arc-recorder: %s: lost (0x%08lx), recording silence until it is back\n", what.c_str(), (unsigned long)hr);
    close();
    down = true;
    retry_at = now_seconds() + 1;
  }
  void retry() {
    if (!down || now_seconds() < retry_at) return;
    if (open()) fprintf(stderr, "arc-recorder: %s: back\n", what.c_str());
    else retry_at = now_seconds() + 1;
  }
};

DWORD pid_of(const std::string& app) {
  if (!app.empty() && app.find_first_not_of("0123456789") == std::string::npos) return (DWORD)std::stoul(app);
  HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  PROCESSENTRY32W e = {sizeof e};
  DWORD pid = 0;
  std::wstring want(app.begin(), app.end());
  for (wchar_t& c : want) c = (wchar_t)towlower(c);
  if (want.size() < 4 || want.substr(want.size() - 4) != L".exe") want += L".exe";
  for (BOOL ok = Process32FirstW(s, &e); ok; ok = Process32NextW(s, &e)) {
    std::wstring n = e.szExeFile;
    for (wchar_t& c : n) c = (wchar_t)towlower(c);
    if (n == want) { pid = e.th32ProcessID; break; }  // (the first; its children are included)
  }
  CloseHandle(s);
  return pid;
}

void wav_header(std::fstream& f, int ch, uint64_t frames) {
  const uint32_t data = (uint32_t)std::min<uint64_t>(frames * 2 * ch, 0xFFFFFFF0u);
  uint8_t h[44];
  auto u32 = [&](int o, uint32_t v) { for (int i = 0; i < 4; ++i) h[o + i] = (uint8_t)(v >> (8 * i)); };
  auto u16 = [&](int o, uint32_t v) { h[o] = (uint8_t)v; h[o + 1] = (uint8_t)(v >> 8); };
  memcpy(h, "RIFF", 4);
  u32(4, 36 + data);
  memcpy(h + 8, "WAVEfmt ", 8);
  u32(16, 16);
  u16(20, 1);
  u16(22, (uint32_t)ch);
  u32(24, kRate);
  u32(28, (uint32_t)(kRate * 2 * ch));
  u16(32, (uint32_t)(2 * ch));
  u16(34, 16);
  memcpy(h + 36, "data", 4);
  u32(40, data);
  const auto end = f.tellp();
  f.seekp(0);
  f.write((const char*)h, 44);
  f.seekp(end);
  f.flush();
}

// a JSON POST to the server on this machine (its port from server.json); the reply's status, 0: not reachable
int post_local(const std::string& path, const std::string& body, std::string& reply) {
  int port = 8090;
  {
    std::ifstream f(fs::path(app_dir()) / "server.json", std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (s.size() >= 3 && (unsigned char)s[0] == 0xEF) s = s.substr(3);  // (PowerShell's UTF-8 BOM)
    Json j;
    if (Json::try_parse(s, j) && j["port"].is_number()) port = (int)j["port"].as_number();
  }
  WSADATA w;
  WSAStartup(MAKEWORD(2, 2), &w);
  SOCKET so = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  sockaddr_in a = {};
  a.sin_family = AF_INET;
  a.sin_port = htons((u_short)port);
  inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  int status = 0;
  if (connect(so, (sockaddr*)&a, sizeof a) == 0) {
    const std::string rq = "POST " + path + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) +
                           "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
                           "\r\nConnection: close\r\n\r\n" + body;
    send(so, rq.data(), (int)rq.size(), 0);
    // (to the end of the body its Content-Length gives: the server keeps the connection open)
    DWORD tmo = 30000;
    setsockopt(so, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tmo, sizeof tmo);
    char b[4096];
    size_t want = std::string::npos;
    for (int n; (n = recv(so, b, sizeof b, 0)) > 0;) {
      reply.append(b, (size_t)n);
      const size_t e = reply.find("\r\n\r\n");
      if (e != std::string::npos && want == std::string::npos) {
        std::string h = reply.substr(0, e);
        for (char& c : h) c = (char)tolower((unsigned char)c);
        const size_t cl = h.find("content-length:");
        want = cl == std::string::npos ? std::string::npos : e + 4 + (size_t)atoll(h.c_str() + cl + 15);
      }
      if (want != std::string::npos && reply.size() >= want) break;
    }
    if (reply.rfind("HTTP/1.1 ", 0) == 0) status = atoi(reply.c_str() + 9);
    const size_t e = reply.find("\r\n\r\n");
    if (e != std::string::npos) reply = reply.substr(e + 4);
  }
  closesocket(so);
  WSACleanup();
  return status;
}

std::atomic<bool> g_stop{false};
BOOL WINAPI on_ctrl(DWORD) {
  g_stop = true;
  return TRUE;
}

int usage() {
  fprintf(stderr,
          "usage: arc-recorder start [--app <name.exe | pid>] [--no-mic] [--title \"...\"] [--seconds N]\n"
          "       arc-recorder stop | status\n"
          "  --app       the meeting app whose sound is the \"Others\" track (Zoom.exe, Teams.exe, chrome.exe, ...);\n"
          "              without it, everything the speakers play\n"
          "  --no-mic    no microphone track (the meeting's sound only)\n"
          "  --seconds   stop by itself after N seconds\n"
          "Writes %%USERPROFILE%%\\arcinferzero\\recordings\\<date> <title>.wav; when it stops, the server transcribes it\n"
          "(the Meetings tab), now or at its next start.\n");
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string cmd = argv[1];
  if (cmd == "stop") {
    HANDLE e = OpenEventW(EVENT_MODIFY_STATE, FALSE, kStopEvent);
    if (!e) { fprintf(stderr, "arc-recorder: no recording runs\n"); return 1; }
    SetEvent(e);
    CloseHandle(e);
    fprintf(stderr, "arc-recorder: stopping\n");
    return 0;
  }
  if (cmd == "status") {
    std::ifstream f(state_file(), std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    HANDLE e = OpenEventW(SYNCHRONIZE, FALSE, kStopEvent);
    if (!e || s.empty()) { printf("not recording\n"); if (e) CloseHandle(e); return 0; }
    CloseHandle(e);
    printf("%s\n", s.c_str());
    return 0;
  }
  if (cmd != "start") return usage();
  std::string app, title = "meeting";
  bool mic = true;
  double limit = 0;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(1); } return argv[++i]; };
    if (a == "--app") app = next();
    else if (a == "--no-mic") mic = false;
    else if (a == "--title") title = next();
    else if (a == "--seconds") limit = atof(next().c_str());
    else return usage();
  }
  HANDLE stop = CreateEventW(nullptr, TRUE, FALSE, kStopEvent);
  if (GetLastError() == ERROR_ALREADY_EXISTS) { fprintf(stderr, "arc-recorder: a recording already runs (arc-recorder stop ends it)\n"); return 1; }
  SetConsoleCtrlHandler(on_ctrl, TRUE);
  CoInitializeEx(nullptr, COINIT_MULTITHREADED);

  std::vector<std::unique_ptr<Track>> tracks;
  if (mic) {
    auto t = std::make_unique<Track>();
    t->kind = Track::Mic;
    t->what = "the microphone";
    tracks.push_back(std::move(t));
  }
  {
    auto t = std::make_unique<Track>();
    if (app.empty()) {
      t->kind = Track::Speakers;
      t->what = "the speakers' sound";
    } else {
      t->kind = Track::Process;
      t->pid = pid_of(app);
      t->what = app + "'s sound";
      if (!t->pid) { fprintf(stderr, "arc-recorder: %s is not running\n", app.c_str()); return 1; }
    }
    tracks.push_back(std::move(t));
  }
  for (auto& t : tracks)
    if (!t->open()) {
      fprintf(stderr, "arc-recorder: cannot open %s%s\n", t->what.c_str(),
              t->kind == Track::Process ? " (process loopback needs Windows 10 2004 or later)" : "");
      return 1;
    }

  // the file: <data>\recordings\<yyyy-mm-dd hh.mm> <title>.wav
  std::string clean;
  for (unsigned char c : title)
    if (c >= 0x20 && !strchr("<>:\"/\\|?*", c)) clean += (char)c;
  if (clean.empty()) clean = "meeting";
  if (clean.size() > 80) clean.resize(80);
  const std::time_t tt = std::time(nullptr);
  std::tm tm{};
  localtime_s(&tm, &tt);
  char when[32];
  std::strftime(when, sizeof when, "%Y-%m-%d %H.%M", &tm);
  const fs::path dir = fs::path(app_dir()) / "recordings";
  std::error_code ec;
  fs::create_directories(dir, ec);
  const fs::path wav = dir / fs::path(std::string(when) + " " + clean + ".wav");
  std::fstream f(wav, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
  if (!f) { fprintf(stderr, "arc-recorder: cannot write %s\n", wav.string().c_str()); return 1; }
  const int ch = (int)tracks.size();
  wav_header(f, ch, 0);
  {
    Json s = Json::object();
    s.set("pid", (double)GetCurrentProcessId());
    s.set("path", wav.string());
    s.set("started", (double)tt);
    s.set("app", app.empty() ? "speakers" : app);
    s.set("microphone", mic);
    std::ofstream(state_file(), std::ios::binary) << s.dump();
  }
  fprintf(stderr, "arc-recorder: recording %s%s into %s (arc-recorder stop, or Ctrl+C, ends it)\n", mic ? "the microphone + " : "",
          tracks.back()->what.c_str(), wav.string().c_str());

  // the loop: the packets drained as their events come; every 100 ms what is 0.3 s old written (late packets placed
  // first); the header put right every second
  const double t0 = now_seconds();
  uint64_t written = 0;
  double last_header = t0;
  std::vector<HANDLE> evs;
  for (auto& t : tracks) evs.push_back(t->ev);
  std::vector<int16_t> buf;
  bool finishing = false;
  for (;;) {
    WaitForMultipleObjects((DWORD)evs.size(), evs.data(), FALSE, 100);
    for (auto& t : tracks) {
      t->drain(t0);
      t->retry();
    }
    const double now = now_seconds();
    if (g_stop || WaitForSingleObject(stop, 0) == WAIT_OBJECT_0 || (limit > 0 && now - t0 >= limit)) finishing = true;
    const int64_t upto = (int64_t)((now - t0 - (finishing ? 0 : 0.3)) * kRate);
    if (upto > (int64_t)written) {
      const size_t n = (size_t)(upto - (int64_t)written);
      buf.assign(n * ch, 0);
      for (int c = 0; c < ch; ++c) {
        std::vector<float>& r = tracks[c]->ring;
        for (size_t i = 0; i < n; ++i) {
          float& v = r[(size_t)((written + i) % r.size())];
          const float x = std::max(-1.0f, std::min(1.0f, v));
          buf[i * ch + c] = (int16_t)std::lround(x < 0 ? x * 32768 : x * 32767);
          v = 0;  // (the slot read: silence until a packet writes it again)
        }
      }
      f.write((const char*)buf.data(), (std::streamsize)(buf.size() * 2));
      written += n;
      if (!f) { fprintf(stderr, "arc-recorder: cannot write the recording (disk full?)\n"); finishing = true; }
    }
    if (now - last_header >= 1 || finishing) {
      wav_header(f, ch, written);
      last_header = now;
    }
    if (finishing) break;
  }
  for (auto& t : tracks) t->close();
  f.close();
  fs::remove(state_file(), ec);
  const double secs = (double)written / kRate;
  fprintf(stderr, "arc-recorder: %.0f s recorded (%llu packets)\n", secs, (unsigned long long)(tracks[0]->packets + tracks.back()->packets));

  // to the server: the transcript library (the Meetings tab), or a note for its next start
  Json m = Json::object(), sp = Json::array();
  if (mic) sp.push("Me");
  sp.push("Others");
  m.set("path", wav.string());
  m.set("name", fs::path(wav).filename().string());
  m.set("kind", "meeting");
  m.set("tracks", (double)ch);
  m.set("speakers", sp);
  m.set("seconds", secs);
  m.set("complete", true);
  {
    fs::path side = wav;
    side.replace_extension(".json");
    std::ofstream(side, std::ios::binary) << m.dump();
  }
  std::string reply;
  const int st = post_local("/v1/transcripts/import", m.dump(), reply);
  if (st == 200) fprintf(stderr, "arc-recorder: handed to the server: the Meetings tab transcribes it now\n");
  else fprintf(stderr, "arc-recorder: the server %s: it takes the recording at its next start\n",
               st ? ("said " + std::to_string(st) + " " + reply.substr(0, 200)).c_str() : "is not running");
  CloseHandle(stop);
  CoUninitialize();
  return 0;
}
