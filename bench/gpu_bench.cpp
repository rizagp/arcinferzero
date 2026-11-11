// What this GPU really does, measured, next to what its kind of GPU should do on paper: the matrix engines (XMX:
// int8, f16, bf16 DPAS, in the sub-group size the GPU's DPAS takes), the vector engines (FP32 and FP16 multiply-add,
// int8 DP4a) and memory read bandwidth. To check the vendors' figures (Intel: the Arc 140T 77 int8 TOPS, a Pro B60
// ~197; TechPowerUp's database counts the 140T at twice that) and to see how far the engine's kernels are from them.
//   bin\gpu_bench.exe [--seconds 1.5] [--out DIR] [--no-html]
//   bin\gpu_bench.exe --page [--out DIR]       only the page again, from the results in DIR
// Writes DIR\<gpu>.json and DIR\gpu-benchmark.html (default DIR: %USERPROFILE%\.inferzero\bench\gpu): the page shows
// every result in DIR side by side, so results from other machines (their .json) dropped there are compared too.
//
// Each compute test keeps 8-16 independent accumulators in registers and repeats one instruction on them: no memory
// traffic in the loop, so the time is the instructions'. A launch fills every thread of every core; it is repeated
// for --seconds and the best launch counts. A multiply-add counts as two operations (Intel's convention).
// On paper: EUs x the driver's maximum clock x the per-EU rate of the GPU's kind, the rates that give Intel's figures
// (Xe-LPG Plus / Xe-HPG: int8 DPAS 256 per EU and clock, f16 / bf16 128, FP32 16; Xe2 / Xe3: 512, 256, 32). A GPU
// held below its maximum clock (power, heat, a laptop on battery) measures below paper by that much.
#include "../zero/clrt.h"
#include "../src/app_dir.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const char* kSrc = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define NACC 8
// x: operands (read once); out: written only for a value no input gives, so the compiler keeps every instruction
#if defined(K_I8_16)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void probe(__global const int* x, int iters, __global int* out) {
  const int l = get_sub_group_local_id();
  const short8 a = as_short8(vload4(0, x + 4 * l));
  const int8 b = vload8(0, x + 64 + 8 * l);
  int8 c[NACC];
  for (int i = 0; i < NACC; ++i) c[i] = (int8)(i);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) c[i] = intel_sub_group_i8_i8_matrix_mad_k32(a, b, c[i]);
  }
  int s = 0;
  for (int i = 0; i < NACC; ++i) s += c[i].s0 ^ c[i].s1 ^ c[i].s2 ^ c[i].s3 ^ c[i].s4 ^ c[i].s5 ^ c[i].s6 ^ c[i].s7;
  if (s == 0x7f3e9a15) out[get_global_id(0)] = s;
}
#elif defined(K_I8_8)
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void probe(__global const int* x, int iters, __global int* out) {
  const int l = get_sub_group_local_id();
  const int8 a = vload8(0, x + 8 * l), b = vload8(0, x + 64 + 8 * l);
  int8 c[NACC];
  for (int i = 0; i < NACC; ++i) c[i] = (int8)(i);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) c[i] = intel_sub_group_i8_i8_matrix_mad_k32(a, b, c[i]);
  }
  int s = 0;
  for (int i = 0; i < NACC; ++i) s += c[i].s0 ^ c[i].s1 ^ c[i].s2 ^ c[i].s3 ^ c[i].s4 ^ c[i].s5 ^ c[i].s6 ^ c[i].s7;
  if (s == 0x7f3e9a15) out[get_global_id(0)] = s;
}
#elif defined(K_F16_16) || defined(K_BF16_16)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void probe(__global const int* x, int iters, __global int* out) {
  const int l = get_sub_group_local_id();
  const short8 a = as_short8(vload4(0, x + 4 * l) & (int4)(0x3C003C00));  // (small values: nothing overflows)
  const int8 b = vload8(0, x + 64 + 8 * l) & (int8)(0x24002400);
  float8 c[NACC];
  for (int i = 0; i < NACC; ++i) c[i] = (float8)(i);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i)
#if defined(K_F16_16)
      c[i] = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c[i]);
#else
      c[i] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c[i]);
#endif
  }
  float s = 0.0f;
  for (int i = 0; i < NACC; ++i) s += c[i].s0 + c[i].s1 + c[i].s2 + c[i].s3 + c[i].s4 + c[i].s5 + c[i].s6 + c[i].s7;
  if (s == -1234.5f) out[get_global_id(0)] = (int)s;
}
#elif defined(K_F16_8) || defined(K_BF16_8)
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void probe(__global const int* x, int iters, __global int* out) {
  const int l = get_sub_group_local_id();
  const int8 a = vload8(0, x + 8 * l) & (int8)(0x3C003C00), b = vload8(0, x + 64 + 8 * l) & (int8)(0x24002400);
  float8 c[NACC];
  for (int i = 0; i < NACC; ++i) c[i] = (float8)(i);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NACC; ++i)
#if defined(K_F16_8)
      c[i] = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c[i]);
#else
      c[i] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c[i]);
#endif
  }
  float s = 0.0f;
  for (int i = 0; i < NACC; ++i) s += c[i].s0 + c[i].s1 + c[i].s2 + c[i].s3 + c[i].s4 + c[i].s5 + c[i].s6 + c[i].s7;
  if (s == -1234.5f) out[get_global_id(0)] = (int)s;
}
#elif defined(K_FP32)
// the vector engines: 8 independent float4 multiply-add chains per work-item
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void probe(__global const int* x, int iters, __global int* out) {
  const float m = 0.999f + 1e-7f * (float)x[0], d = 1e-6f * (float)x[1];
  float4 c[8];
  for (int i = 0; i < 8; ++i) c[i] = (float4)((float)(i + get_global_id(0)));
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = fma(c[i], (float4)(m), (float4)(d));
  }
  float s = 0.0f;
  for (int i = 0; i < 8; ++i) s += c[i].x + c[i].y + c[i].z + c[i].w;
  if (s == -1234.5f) out[get_global_id(0)] = (int)s;
}
#elif defined(K_FP16)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void probe(__global const int* x, int iters, __global int* out) {
  const half m = (half)(0.999f + 1e-7f * (float)x[0]), d = (half)(1e-4f * (float)x[1]);
  half8 c[8];
  for (int i = 0; i < 8; ++i) c[i] = (half8)((half)(i & 7));
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = fma(c[i], (half8)(m), (half8)(d));
  }
  float s = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const float8 f = convert_float8(c[i]);
    s += f.s0 + f.s1 + f.s2 + f.s3 + f.s4 + f.s5 + f.s6 + f.s7;
  }
  if (s == -1234.5f) out[get_global_id(0)] = (int)s;
}
#elif defined(K_DP4A)
// int8 on the vector engines: one 4 x int8 dot product with accumulate per lane and instruction, 16 chains
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void probe(__global const int* x, int iters, __global int* out) {
  const int l = get_sub_group_local_id();
  const int a = x[l], b = x[64 + l];
  int c[16];
  for (int i = 0; i < 16; ++i) c[i] = i;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < 16; ++i) c[i] = dot_acc_sat_4x8packed_ss_int(a + i, b, c[i]);
  }
  int s = 0;
  for (int i = 0; i < 16; ++i) s ^= c[i];
  if (s == 0x7f3e9a15) out[get_global_id(0)] = s;
}
#elif defined(K_READ)
// memory read bandwidth: 8 independent 16-byte loads in flight per work-item, each step contiguous over the grid
__kernel void probe(__global const uint4* x, int n4, __global int* out) {
  uint4 a[8];
  for (int u = 0; u < 8; ++u) a[u] = (uint4)(0);
  const int g = get_global_size(0);
  for (int i = get_global_id(0); i + 7 * g < n4; i += 8 * g)
    for (int u = 0; u < 8; ++u) a[u] += x[i + u * g];
  for (int u = 1; u < 8; ++u) a[0] += a[u];
  if ((a[0].x ^ a[0].y ^ a[0].z ^ a[0].w) == 0x9e3779b9u) out[0] = 1;
}
#endif
)CLC";

using clk = std::chrono::steady_clock;

struct Test {
  const char* id;       // JSON key
  const char* label;
  const char* def;
  int lanes;            // sub-group size
  double ops_per_step;  // operations per work-item ... per sub-group and loop step (a multiply-add = 2)
  const char* unit;
  bool dpas;
  int per_eu;           // on paper: operations per EU and clock (0: none)
  double measured = 0, paper = 0;
  std::string note;
};

static std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if ((unsigned char)c < 0x20) o += ' ';
    else o += c;
  }
  return o + "\"";
}
// the page: a comparison table of every result in dir, then a card per GPU (the layout of a spec database's page)
static void write_page(const std::string& dir) {
  namespace fs = std::filesystem;
  std::vector<std::string> blobs;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (e.path().extension() != ".json") continue;
    std::ifstream f(e.path(), std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    if (s.size() > 2 && s.front() == '{' && s.back() == '}' && s.find("</") == std::string::npos) blobs.push_back(s);
  }
  std::string data = "[";
  for (size_t i = 0; i < blobs.size(); ++i) data += (i ? ",\n" : "\n") + blobs[i];
  data += "\n]";
  const std::string page = std::string(R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU benchmark</title>
<style>
:root{--bg:#f4f5f8;--card:#fff;--ink:#1d2330;--mute:#6b7385;--line:#e3e6ee;--head:#3d5585;--headink:#fff;--good:#1f7a4d;--warn:#b26a00;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--card:#1d2129;--ink:#e6e9ef;--mute:#9aa3b5;--line:#2c323d;--head:#33497a;--headink:#f2f4f8;--good:#5cc893;--warn:#e0a240;--bad:#ef7a72}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:1240px;margin:0 auto;padding:20px 16px 48px}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--mute);margin:0 0 18px}
.tbl{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:6px;margin-bottom:28px}
table{border-collapse:collapse;width:100%;min-width:560px}th,td{padding:7px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}thead th{background:var(--head);color:var(--headink);font-weight:600}
td small{color:var(--mute);margin-left:6px}.sec td{background:var(--bg);font-weight:600;text-align:left;color:var(--mute)}
h2{font-size:18px;margin:26px 0 10px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:16px;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.card h3{margin:0;background:var(--head);color:var(--headink);font-size:16px;text-align:center;padding:9px 8px}
.row{display:flex;gap:12px;padding:8px 14px;border-bottom:1px solid var(--line)}.row:last-child{border-bottom:0}
.row b{flex:0 0 46%;text-align:right;font-weight:600}.row span{flex:1}.row span small{display:block;color:var(--mute)}
.pc{font-weight:600}.g{color:var(--good)}.w{color:var(--warn)}.r{color:var(--bad)}
.notes{background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden;margin-top:16px}
.notes h3{margin:0;background:var(--head);color:var(--headink);font-size:16px;text-align:center;padding:9px 8px}
.notes pre{margin:0;padding:12px 14px;font:13px/1.6 ui-monospace,Consolas,monospace;white-space:pre-wrap}
</style></head><body><main>
<h1>GPU benchmark</h1>
<p class="sub">Measured by <code>bin\gpu_bench.exe</code> (inferzero) next to each GPU's figures on paper: its EUs x maximum clock x
the per-EU rate of its kind (the rates behind Intel's numbers). A multiply-add counts as two operations. Every result
file (<code>.json</code>) in this folder is shown.</p>
<div class="tbl"><table id="cmp"></table></div>
<div id="cards"></div>
</main><script>
const R=)HTML") + data + R"HTML(;
const fmt=(v,u)=>v>=100?v.toFixed(0)+' '+u:v>=10?v.toFixed(1)+' '+u:v.toFixed(2)+' '+u;
const pcls=p=>p>=70?'g':p>=40?'w':'r';
const label=r=>(r.short||r.name)+(r.family?' \u00b7 '+r.family:'');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function val(t){if(!t||!(t.measured>0))return t&&t.note?'<small>'+esc(t.note)+'</small>':'&ndash;';
 const p=t.paper>0?Math.round(100*t.measured/t.paper):null;
 return esc(fmt(t.measured,t.unit))+(p!==null?' <small class="pc '+pcls(p)+'">'+p+'%</small>':'');}
const rows=[['Matrix (XMX)',null],['int8','i8'],['f16','f16'],['bf16','bf16'],['Vector',null],['FP32','fp32'],['FP16','fp16'],['int8 DP4a','dp4a'],['Memory',null],['Read bandwidth','read']];
let h='<thead><tr><th></th>'+R.map(r=>'<th>'+esc(label(r))+'</th>').join('')+'</tr></thead><tbody>';
for(const [lab,id] of rows){ if(!id){h+='<tr class="sec"><td colspan="'+(R.length+1)+'">'+lab+'</td></tr>';continue;}
 h+='<tr><td>'+lab+'</td>'+R.map(r=>'<td>'+val(r.tests[id])+'</td>').join('')+'</tr>';}
h+='<tr class="sec"><td colspan="'+(R.length+1)+'">Device</td></tr>';
for(const [lab,f] of [['Architecture',r=>r.arch],['EUs / Xe-cores',r=>r.eus+' / '+r.cores],['Maximum clock',r=>r.mhz+' MHz'],['Measured',r=>r.date]])
 h+='<tr><td>'+lab+'</td>'+R.map(r=>'<td>'+esc(f(r))+'</td>').join('')+'</tr>';
document.getElementById('cmp').innerHTML=h+'</tbody>';
const card=(t,rs)=>'<div class="card"><h3>'+t+'</h3>'+rs.map(([k,v])=>'<div class="row"><b>'+k+':</b><span>'+v+'</span></div>').join('')+'</div>';
const paper=t=>t&&t.paper>0?'<small>paper '+esc(fmt(t.paper,t.unit))+'</small>':'';
let c='';
for(const r of R){
 c+='<h2>'+esc(label(r))+'</h2><div class="grid">';
 c+=card('Graphics Processor',[['GPU name',esc(r.name)],['Device ID',esc(r.device)],['Architecture',esc(r.arch+(r.family?' ('+r.family+')':''))],['Driver',esc(r.driver)],['Memory',esc(r.memory)]]);
 c+=card('Render Config',[['Execution units',r.eus],['Xe-cores',r.cores],['XMX engines',r.xmx_engines||'none'],['DPAS sub-group',r.dpas_lanes?r.dpas_lanes+' lanes':'none'],['Maximum clock',r.mhz+' MHz']]);
 c+=card('Matrix Performance',[['INT8',val(r.tests.i8)+paper(r.tests.i8)],['FP16',val(r.tests.f16)+paper(r.tests.f16)],['BF16',val(r.tests.bf16)+paper(r.tests.bf16)]]);
 c+=card('Vector Performance',[['FP32',val(r.tests.fp32)+paper(r.tests.fp32)],['FP16',val(r.tests.fp16)+paper(r.tests.fp16)],['INT8 (DP4a)',val(r.tests.dp4a)+paper(r.tests.dp4a)]]);
 c+=card('Memory',[['Read bandwidth',val(r.tests.read)],['Shared with CPU',r.integrated?'yes':'no']]);
 c+='</div>';
 const notes=['Architecture: '+(r.arch||'?')+(r.family?' ('+r.family+')':''),'IP version: '+(r.ip||'unknown'),
  'Xe-Cores: '+(r.core_gen||'?'),'XMX Engines: '+(r.xmx_gen||'?'),
  'DPAS: '+(r.dpas_lanes?r.dpas_lanes+'-lane sub-groups':'none (DP4a on the vector engines)'),
  'Threads per EU: '+(r.threads_per_eu||'?'),'Measured: '+r.date+', driver '+(r.driver||'?')];
 c+='<div class="notes"><h3>'+esc((r.family||r.short||r.name)+' GPU Notes')+'</h3><pre>'+esc(notes.join('\n'))+'</pre></div>';}
document.getElementById('cards').innerHTML=c;
</script></body></html>
)HTML";
  std::ofstream(fs::path(dir) / "gpu-benchmark.html", std::ios::binary) << page;
}

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  double seconds = 1.5;
  bool html = true, page_only = false;
  std::string out = (fs::path(app_dir()) / "bench" / "gpu").string();
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--seconds" && i + 1 < argc) seconds = atof(argv[++i]);
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--no-html") html = false;
    else if (a == "--page") page_only = true;
    else { fprintf(stderr, "usage: gpu_bench [--seconds 1.5] [--out DIR] [--no-html] [--page]\n"); return 2; }
  }
  std::error_code ec;
  fs::create_directories(out, ec);
  if (page_only) {
    write_page(out);
    printf("%s\n", (fs::path(out) / "gpu-benchmark.html").string().c_str());
    return 0;
  }
  ClRt cl;
  if (!cl.init()) return 1;
  char name[256] = {}, drv[128] = {};
  cl_uint eus = 0, mhz = 0;
  cl_ulong mem = 0;
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102B, sizeof(name) - 1, name, nullptr);  // CL_DEVICE_NAME
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102D, sizeof(drv) - 1, drv, nullptr);    // CL_DRIVER_VERSION
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x1002, sizeof(eus), &eus, nullptr);       // CL_DEVICE_MAX_COMPUTE_UNITS
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x100C, sizeof(mhz), &mhz, nullptr);       // CL_DEVICE_MAX_CLOCK_FREQUENCY
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x101F, sizeof(mem), &mem, nullptr);       // CL_DEVICE_GLOBAL_MEM_SIZE
  const bool igpu = zrt::integrated();
  // the GPU's IP version and layout (cl_intel_device_attribute_query: CL_DEVICE_IP_VERSION_INTEL 0x4250, its GMD ID
  // architecture.release.revision; threads per EU 0x4255). 0 where the driver does not answer: then the kind from what it offers (DPAS, 8-lane sub-groups)
  cl_uint ipv = 0, thr = 0;
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x4250, sizeof(ipv), &ipv, nullptr);
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x4255, sizeof(thr), &thr, nullptr);
  const int ip_arch = (int)(ipv >> 22), ip_rel = (int)((ipv >> 14) & 0xFF), ip_rev = (int)(ipv & 0x3F);
  const bool xe2 = cl.xmx && !cl.sg8, hpg = cl.xmx && cl.sg8;
  std::string arch = xe2 ? "Xe2 / Xe3" : hpg ? (igpu ? "Xe-LPG Plus" : "Xe-HPG") : "Xe-LPG / Xe-LP";
  std::string family, core_gen = xe2 ? "2nd or 3rd generation" : "1st generation", xmx_gen = cl.xmx ? core_gen : "none";
  if (ip_arch == 12 && (ip_rel == 70 || ip_rel == 71)) { arch = "Xe-LPG"; family = "Meteor Lake"; core_gen = "1st generation"; }
  else if (ip_arch == 12 && ip_rel == 74) { arch = "Xe-LPG Plus"; family = "Arrow Lake H"; core_gen = xmx_gen = "1st generation"; }
  else if (ip_arch == 12 && ip_rel >= 55 && ip_rel <= 57) { arch = "Xe-HPG"; family = "Alchemist"; core_gen = xmx_gen = "1st generation"; }
  else if (ip_arch == 12 && ip_rel < 50) { arch = "Xe-LP"; family = "Tiger Lake / Alder Lake"; core_gen = "Xe-LP (EUs)"; }
  else if (ip_arch == 20 && ip_rel <= 2) { arch = "Xe2-HPG"; family = "Battlemage"; core_gen = xmx_gen = "2nd generation"; }
  else if (ip_arch == 20) { arch = "Xe2-LPG"; family = "Lunar Lake"; core_gen = xmx_gen = "2nd generation"; }
  else if (ip_arch == 30) { arch = "Xe3-LPG"; family = "Panther Lake"; core_gen = xmx_gen = "3rd generation"; }
  if (!cl.xmx) xmx_gen = "none";
  char ipt[32] = "unknown";
  if (ipv) snprintf(ipt, sizeof ipt, "%d.%d.%d", ip_arch, ip_rel, ip_rev);
  // (an Xe-core: 16 EUs on Xe-LPG / Xe-LPG Plus / Xe-HPG, 8 on Xe2 / Xe3; the driver's sub-slice is 8 EUs
  //  on Xe-LPG, half a core, so the cores come from the kind)
  const int eu_per_core = ip_arch >= 20 || (!ipv && xe2) ? 8 : 16, dpas_lanes = xe2 ? 16 : hpg ? 8 : 0;
  const int vec = xe2 ? 32 : 16;  // FP32 operations per EU and clock (8 or 16 lanes, multiply-add = 2)
  ze_device_properties_t dp{};
  dp.stype = ZE_STYPE_DEVICE_PROPERTIES;
  zeDeviceGetProperties(zrt::D().dev, &dp);
  char devid[16];
  snprintf(devid, sizeof devid, "0x%04X", dp.deviceId);
  char memtxt[96];
  snprintf(memtxt, sizeof memtxt, igpu ? "%.1f GiB, shared with the CPU" : "%.1f GiB", mem / 1073741824.0);
  printf("GPU: %s (%s), %s%s%s (IP %s), %u EUs (%u Xe-cores), %u threads per EU, maximum clock %u MHz, %s, driver %s\n",
         name, devid, arch.c_str(), family.empty() ? "" : ", ", family.c_str(), ipt, eus, eus / eu_per_core, thr, mhz,
         memtxt, drv);

  const int L = dpas_lanes ? dpas_lanes : 16;
  std::vector<Test> tests = {
      {"i8", "INT8 (XMX)", L == 8 ? "K_I8_8" : "K_I8_16", L, 2.0 * 8 * L * 32, "TOPS", true, xe2 ? 512 : 256},
      {"f16", "FP16 (XMX)", L == 8 ? "K_F16_8" : "K_F16_16", L, 2.0 * 8 * L * 16, "TFLOPS", true, xe2 ? 256 : 128},
      {"bf16", "BF16 (XMX)", L == 8 ? "K_BF16_8" : "K_BF16_16", L, 2.0 * 8 * L * 16, "TFLOPS", true, xe2 ? 256 : 128},
      {"fp32", "FP32 (vector)", "K_FP32", 16, 2.0 * 4 * 16, "TFLOPS", false, vec},
      {"fp16", "FP16 (vector)", "K_FP16", 16, 2.0 * 8 * 16, "TFLOPS", false, 2 * vec},
      {"dp4a", "INT8 DP4a (vector)", "K_DP4A", 16, 2.0 * 4 * 16, "TOPS", false, 4 * vec},
  };
  std::vector<int> host(256);
  for (int i = 0; i < 256; ++i) host[i] = (int)((0x01020304u * (uint32_t)(i + 1)) ^ 0x5a5a5a5au) & 0x3f3f3f3f;
  const int groups = std::max(256u, eus * 16), iters = 2048;  // (16 sub-groups of 4 per EU: every thread slot filled)
  cl_mem bx = cl.buffer(CL_MEM_READ_WRITE, host.size() * 4), bo = cl.buffer(CL_MEM_READ_WRITE, (size_t)groups * 64 * 4);
  CL_CHECK(clEnqueueWriteBuffer(cl.queue, bx, CL_TRUE, 0, host.size() * 4, host.data(), 0, nullptr, nullptr));
  cl.soft_fail = true;
  auto timed = [&](cl_kernel k, ze_group_count_t gc) {
    auto launch = [&] {
      const auto t0 = clk::now();
      ZE_CHECK(zeCommandListAppendLaunchKernel(zrt::D().cl, k->k, &gc, nullptr, 0, nullptr));
      zrt::sync();
      return std::chrono::duration<double>(clk::now() - t0).count();
    };
    launch();  // (the binary's upload, the clocks ramping)
    double best = 1e30, spent = 0;
    while (spent < seconds) { const double t = launch(); best = std::min(best, t); spent += t; }
    return best;
  };
  printf("\n%-22s %14s %14s %8s\n", "", "measured", "on paper", "share");
  for (Test& t : tests) {
    t.paper = t.per_eu ? (double)t.per_eu * eus * mhz * 1e6 / 1e12 : 0;
    if (t.dpas && !cl.xmx) { t.note = "no XMX"; t.paper = 0; printf("%-22s %14s\n", t.label, "no XMX"); continue; }
    try {
      cl_program p = cl.build(kSrc, std::string("-cl-std=CL3.0 -D") + t.def);
      cl_kernel k = cl.kernel(p, "probe");
      int it = iters;
      set_args(k, 0, bx, it, bo);
      ZE_CHECK(zeKernelSetGroupSize(k->k, 4 * t.lanes, 1, 1));
      const int chains = std::string(t.def) == "K_DP4A" ? 16 : 8;
      const double best = timed(k, ze_group_count_t{(uint32_t)groups, 1, 1});
      // (groups x 4 sub-groups; ops_per_step per sub-group: a DPAS 2 * 8 * N * K, a vector instruction 2 * width * lanes)
      t.measured = (double)groups * 4 * t.ops_per_step * chains * iters / best / 1e12;
    } catch (const std::exception& e) {
      t.note = "not available";
      printf("%-22s %14s  (%s)\n", t.label, "not available", e.what());
      continue;
    }
    printf("%-22s %9.2f %-5s %9.2f %-5s %7.0f%%\n", t.label, t.measured, t.unit, t.paper, t.unit,
           t.paper > 0 ? 100.0 * t.measured / t.paper : 0.0);
  }
  // memory read bandwidth: a buffer larger than every cache, read over and over
  double gbs = 0;
  {
    const size_t bytes = (size_t)std::min<cl_ulong>(mem / 4, 1ull << 30) / 4096 * 4096;
    try {
      cl_program p = cl.build(kSrc, "-cl-std=CL3.0 -DK_READ");
      cl_kernel k = cl.kernel(p, "probe");
      cl_mem db = cl.buffer(CL_MEM_READ_WRITE, bytes);
      const uint32_t one = 0x01010101u;
      CL_CHECK(clEnqueueFillBuffer(cl.queue, db, &one, 4, 0, bytes, 0, nullptr, nullptr));
      clFinish(cl.queue);
      int n4 = (int)(bytes / 16);
      set_args(k, 0, db, n4, bo);
      ZE_CHECK(zeKernelSetGroupSize(k->k, 256, 1, 1));
      const double best = timed(k, ze_group_count_t{2048, 1, 1});
      gbs = bytes / best / 1e9;
      printf("%-22s %9.1f GB/s\n", "Memory read", gbs);
    } catch (const std::exception& e) {
      printf("%-22s %14s  (%s)\n", "Memory read", "failed", e.what());
    }
  }

  // ---- the result file: one per GPU (its name and device id), replaced by a newer run
  std::string slug;
  for (char c : std::string(name) + "-" + devid) slug += isalnum((unsigned char)c) ? (char)tolower((unsigned char)c) : '-';
  while (slug.find("--") != std::string::npos) slug.erase(slug.find("--"), 1);
  char date[32];
  const time_t now = time(nullptr);
  strftime(date, sizeof date, "%Y-%m-%d", localtime(&now));
  std::string shortname = name;
  for (const char* cut : {"Intel(R) ", "(R)", "(TM)", " Graphics", " GPU"}) {
    for (size_t q; (q = shortname.find(cut)) != std::string::npos;) shortname.erase(q, strlen(cut));
  }
  std::ostringstream js;
  js << "{\"name\":" << jstr(name) << ",\"short\":" << jstr(shortname) << ",\"device\":" << jstr(devid)
     << ",\"arch\":" << jstr(arch) << ",\"family\":" << jstr(family) << ",\"ip\":" << jstr(ipt)
     << ",\"core_gen\":" << jstr(core_gen) << ",\"xmx_gen\":" << jstr(xmx_gen) << ",\"threads_per_eu\":" << thr
     << ",\"driver\":" << jstr(drv) << ",\"memory\":" << jstr(memtxt)
     << ",\"integrated\":" << (igpu ? "true" : "false") << ",\"eus\":" << eus << ",\"cores\":" << eus / eu_per_core
     << ",\"xmx_engines\":" << (cl.xmx ? eus : 0) << ",\"dpas_lanes\":" << dpas_lanes << ",\"mhz\":" << mhz
     << ",\"date\":" << jstr(date) << ",\"tests\":{";
  for (size_t i = 0; i < tests.size(); ++i) {
    const Test& t = tests[i];
    char b[64];
    js << (i ? "," : "") << "\"" << t.id << "\":{\"measured\":";
    snprintf(b, sizeof b, "%.4f", t.measured); js << b << ",\"paper\":";
    snprintf(b, sizeof b, "%.4f", t.paper); js << b << ",\"unit\":" << jstr(t.unit) << ",\"note\":" << jstr(t.note) << "}";
  }
  char b[64];
  snprintf(b, sizeof b, "%.2f", gbs);
  js << ",\"read\":{\"measured\":" << b << ",\"paper\":0,\"unit\":\"GB/s\",\"note\":\"\"}}}\n";
  const fs::path jp = fs::path(out) / (slug + ".json");
  std::ofstream(jp, std::ios::binary) << js.str();
  printf("\nresult: %s\n", jp.string().c_str());
  if (html) {
    write_page(out);
    printf("page: %s\n", (fs::path(out) / "gpu-benchmark.html").string().c_str());
  }
  printf("\nOn paper: EUs x maximum clock x the per-EU rate of the GPU's kind (Intel's convention: a multiply-add is\n"
         "two operations). A share well below 100 %% means a clock held below its maximum, or fewer engines than paper.\n");
  return 0;
}
