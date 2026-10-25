// arcinferzero NPU experiment: a minimal OpenVINO IR v11 writer (the model format the NPU driver's compiler takes,
// ZE_GRAPH_FORMAT_NGRAPH_LITE) and its serialization, enough for the matrix-product graphs of zero/npu/.
#pragma once
#include "ze_npu.h"
#include <cstring>
#include <string>
#include <vector>

namespace npu {

inline uint16_t f2h(float f) {  // f32 -> f16, round to nearest even (normal range)
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t s = (x >> 16) & 0x8000u;
  const int e = (int)((x >> 23) & 0xff) - 127 + 15;
  const uint32_t m = x & 0x7fffffu;
  if (e <= 0) return (uint16_t)s;
  if (e >= 31) return (uint16_t)(s | 0x7c00u);
  uint32_t h = s | ((uint32_t)e << 10) | (m >> 13);
  const uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
  return (uint16_t)h;
}
inline float h2f(uint16_t h) {
  const uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu;
  uint32_t x;
  if (e == 0) {
    if (m == 0) x = s;
    else {
      int k = 0;
      uint32_t mm = m;
      while (!(mm & 0x400u)) { mm <<= 1; ++k; }
      x = s | ((uint32_t)(113 - k) << 23) | ((mm & 0x3ffu) << 13);
    }
  } else if (e == 31) x = s | 0x7f800000u | (m << 13);
  else x = s | ((e + 112) << 23) | (m << 13);
  float f;
  memcpy(&f, &x, 4);
  return f;
}

// A tensor: its element type ("f16", "i8", "f32") and shape.
struct T { std::string et; std::vector<int> shape; };
inline const char* precision(const std::string& et) {
  return et == "f16" ? "FP16" : et == "f32" ? "FP32" : et == "i8" ? "I8" : et == "u8" ? "U8" : et == "i32" ? "I32" : et == "i64" ? "I64" : "UNSPECIFIED";
}

class Ir {
 public:
  struct Out { int layer, port; T t; };
  Out param(const std::string& name, const T& t) {
    std::string d = "<data shape=\"" + dims_csv(t.shape) + "\" element_type=\"" + t.et + "\"/>";
    return add(name, "Parameter", d, {}, t, name);
  }
  Out const_f16(const std::string& name, const std::vector<int>& shape, const std::vector<float>& v) {
    const size_t off = w_.size();
    for (float x : v) {
      const uint16_t h = f2h(x);
      w_.push_back((uint8_t)(h & 0xff));
      w_.push_back((uint8_t)(h >> 8));
    }
    const std::string d = "<data element_type=\"f16\" shape=\"" + dims_csv(shape) + "\" offset=\"" + std::to_string(off) +
                          "\" size=\"" + std::to_string(v.size() * 2) + "\"/>";
    return add(name, "Const", d, {}, T{"f16", shape}, "");
  }
  Out const_i64(const std::string& name, const std::vector<int64_t>& v) {  // a 1-D i64 constant (Reshape's shape)
    const size_t off = w_.size();
    for (int64_t x : v) w_.insert(w_.end(), (const uint8_t*)&x, (const uint8_t*)&x + 8);
    const std::string d = "<data element_type=\"i64\" shape=\"" + std::to_string(v.size()) + "\" offset=\"" + std::to_string(off) +
                          "\" size=\"" + std::to_string(v.size() * 8) + "\"/>";
    return add(name, "Const", d, {}, T{"i64", {(int)v.size()}}, "");
  }
  Out reshape(const std::string& name, const Out& a, const std::vector<int>& shape) {
    std::vector<int64_t> s(shape.begin(), shape.end());
    return add(name, "Reshape", "<data special_zero=\"false\"/>", {a, const_i64(name + "_shape", s)}, T{a.t.et, shape}, "");
  }
  Out convert(const std::string& name, const Out& a, const std::string& to) {
    return add(name, "Convert", "<data destination_type=\"" + to + "\"/>", {a}, T{to, a.t.shape}, "");
  }
  Out multiply(const std::string& name, const Out& a, const Out& b) {  // (numpy broadcast; the result has a's shape)
    return add(name, "Multiply", "<data auto_broadcast=\"numpy\"/>", {a, b}, T{a.t.et, a.t.shape}, "");
  }
  Out add_op(const std::string& name, const Out& a, const Out& b) {
    return add(name, "Add", "<data auto_broadcast=\"numpy\"/>", {a, b}, T{a.t.et, a.t.shape}, "");
  }
  Out matmul_tb(const std::string& name, const Out& a, const Out& b) {  // a [M][K] . b [N][K]^T
    return add(name, "MatMul", "<data transpose_a=\"false\" transpose_b=\"true\"/>", {a, b},
               T{a.t.et, {a.t.shape[0], b.t.shape[0]}}, "");
  }
  // 1x1 convolution, NCHW: x [1][C][H][W] with weights [O][C][1][1] -> [1][O][H][W]
  Out conv1x1(const std::string& name, const Out& x, const Out& w) {
    return add(name, "Convolution",
               "<data strides=\"1,1\" dilations=\"1,1\" pads_begin=\"0,0\" pads_end=\"0,0\" auto_pad=\"explicit\"/>", {x, w},
               T{x.t.et, {1, w.t.shape[0], x.t.shape[2], x.t.shape[3]}}, "");
  }
  Out fake_quantize(const std::string& name, const Out& a, const Out& lo, const Out& hi, int levels) {
    return fake_quantize(name, a, lo, hi, lo, hi, levels);
  }
  // input range [ilo, ihi] in `levels` steps, written out on [olo, ohi]
  Out fake_quantize(const std::string& name, const Out& a, const Out& ilo, const Out& ihi, const Out& olo, const Out& ohi,
                    int levels) {
    return add(name, "FakeQuantize", "<data levels=\"" + std::to_string(levels) + "\" auto_broadcast=\"numpy\"/>",
               {a, ilo, ihi, olo, ohi}, a.t, "");
  }
  void result(const Out& a) { add(a.t.et + "_result", "Result", "", {a}, T{}, "", true); }

  std::string xml() const {
    return "<?xml version=\"1.0\"?>\n<net name=\"arc\" version=\"11\">\n<layers>\n" + layers_ + "</layers>\n<edges>\n" + edges_ +
           "</edges>\n<rt_info/>\n</net>\n";
  }
  const std::vector<uint8_t>& weights() const { return w_; }
  // the driver compiler's input: compiler version, 2 parts, xml size + xml, weights size + weights
  std::vector<uint8_t> serialize(ze_graph_compiler_version_info_t cv) const {
    std::vector<uint8_t> b;
    auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n); };
    const std::string x = xml();
    const uint32_t parts = 2;
    const uint64_t xs = x.size(), ws = w_.size();
    put(&cv, sizeof cv);
    put(&parts, 4);
    put(&xs, 8);
    put(x.data(), x.size());
    put(&ws, 8);
    if (!w_.empty()) put(w_.data(), w_.size());
    return b;
  }

 private:
  int next_ = 0;
  std::string layers_, edges_;
  std::vector<uint8_t> w_;
  static std::string dims_csv(const std::vector<int>& s) {
    std::string r;
    for (size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
    return r;
  }
  static std::string port(int id, const T& t, const std::string& names) {
    std::string r = "<port id=\"" + std::to_string(id) + "\" precision=\"" + precision(t.et) + "\"";
    if (!names.empty()) r += " names=\"" + names + "\"";
    r += ">";
    for (int d : t.shape) r += "<dim>" + std::to_string(d) + "</dim>";
    return r + "</port>";
  }
  Out add(const std::string& name, const char* type, const std::string& data, const std::vector<Out>& in, const T& out,
          const std::string& out_names, bool is_result = false) {
    const int id = next_++;
    std::string l = "<layer id=\"" + std::to_string(id) + "\" name=\"" + name + "\" type=\"" + type + "\" version=\"opset1\">" + data;
    if (!in.empty()) {
      l += "<input>";
      for (size_t i = 0; i < in.size(); ++i) {
        l += port((int)i, in[i].t, "");
        edges_ += "<edge from-layer=\"" + std::to_string(in[i].layer) + "\" from-port=\"" + std::to_string(in[i].port) +
                  "\" to-layer=\"" + std::to_string(id) + "\" to-port=\"" + std::to_string(i) + "\"/>\n";
      }
      l += "</input>";
    }
    const int op = (int)in.size();
    if (!is_result) l += "<output>" + port(op, out, out_names.empty() ? name : out_names) + "</output>";
    layers_ += l + "</layer>\n";
    return Out{id, op, out};
  }
};

}  // namespace npu
