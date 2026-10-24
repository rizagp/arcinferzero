// Renders a GGUF chat template as written, with minja (Google's C++ implementation of the Jinja subset used by
// Hugging Face chat templates, third_party/minja), after two rewrites minja needs (below). Used for model families
// whose template is not implemented natively in chat.h (Gemma 4).
#pragma once
#include "json.h"
#include <minja/chat-template.hpp>
#include <string>

class JinjaChat {
 public:
  JinjaChat(const std::string& source, const std::string& bos, const std::string& eos)
      : t_(join_adjacent_strings(source), bos, eos) {}

  // Jinja2 concatenates adjacent string literals ("a " "b"); minja does not parse them (Google's Gemma 4 template
  // of 2026-07-09 uses them in raise_exception calls). Inside {{ }} / {% %} blocks, `"x" "y"` (same quote, only
  // whitespace between) becomes `"xy"`; text outside the blocks is left alone.
  static std::string join_adjacent_strings(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    bool code = false;
    while (i < s.size()) {
      if (!code) {
        if (s[i] == '{' && i + 1 < s.size() && (s[i + 1] == '{' || s[i + 1] == '%')) code = true;
        out += s[i++];
        continue;
      }
      const char c = s[i];
      if ((c == '}' || c == '%') && i + 1 < s.size() && s[i + 1] == '}') {
        code = false;
        out += c;
        out += s[i + 1];
        i += 2;
        continue;
      }
      // (minja has the string method .replace(old, new) but not the filter: `x | replace(a, b)` becomes
      //  `x.replace(a, b)`, the same in Jinja (a filter binds as tightly as a method call))
      if (c == '|') {
        size_t j = i + 1;
        while (j < s.size() && (s[j] == ' ' || s[j] == '\t')) ++j;
        if (s.compare(j, 7, "replace") == 0) {
          size_t k = j + 7;
          while (k < s.size() && (s[k] == ' ' || s[k] == '\t')) ++k;
          if (k < s.size() && s[k] == '(') {
            out += ".replace(";
            i = k + 1;
            continue;
          }
        }
      }
      // (`tojson(ensure_ascii=False)`: minja's tojson takes no such argument and already writes UTF-8 as is)
      if (c == 't' && s.compare(i, 7, "tojson(") == 0 && (i == 0 || !(isalnum((unsigned char)s[i - 1]) || s[i - 1] == '_'))) {
        size_t k = i + 7;
        auto skip = [&] { while (k < s.size() && (s[k] == ' ' || s[k] == '\t')) ++k; };
        skip();
        if (s.compare(k, 12, "ensure_ascii") == 0) {
          k += 12;
          skip();
          if (k < s.size() && s[k] == '=') {
            ++k;
            skip();
            if (s.compare(k, 5, "False") == 0) {
              k += 5;
              skip();
              if (k < s.size() && s[k] == ')') {
                out += "tojson";
                i = k + 1;
                continue;
              }
            }
          }
        }
      }
      if (c != '"' && c != '\'') {
        out += s[i++];
        continue;
      }
      // a string literal: copy it, then absorb following literals of the same quote
      out += c;
      ++i;
      for (;;) {
        while (i < s.size() && s[i] != c) {
          if (s[i] == '\\' && i + 1 < s.size()) out += s[i++];
          out += s[i++];
        }
        size_t j = i + 1;  // past the closing quote
        while (j < s.size() && (s[j] == ' ' || s[j] == '\t' || s[j] == '\n' || s[j] == '\r')) ++j;
        if (i < s.size() && j < s.size() && s[j] == c && j > i + 1) {
          i = j + 1;  // drop the closing quote, the whitespace and the next opening quote
          continue;
        }
        if (i < s.size()) out += s[i++];  // closing quote
        break;
      }
    }
    return out;
  }

  // messages / tools in OpenAI form. Tool-call arguments sent as JSON strings are turned into objects (templates
  // iterate over them). `enable_thinking` and `reasoning_effort` are passed as template variables.
  std::string render(const Json& messages, const Json& tools, bool add_generation_prompt, bool enable_thinking,
                     const std::string& reasoning_effort = "") const {
    minja::chat_template_inputs in;
    in.messages = nlohmann::ordered_json::parse(messages.dump());
    for (auto& m : in.messages) {
      if (!m.is_object() || !m.contains("tool_calls") || !m["tool_calls"].is_array()) continue;
      for (auto& tc : m["tool_calls"]) {
        auto& fn = tc.contains("function") ? tc["function"] : tc;
        if (fn.contains("arguments") && fn["arguments"].is_string()) {
          const std::string a = fn["arguments"].get<std::string>();
          if (nlohmann::ordered_json::accept(a)) fn["arguments"] = nlohmann::ordered_json::parse(a);
        }
      }
    }
    if (tools.is_array() && tools.size() > 0) in.tools = nlohmann::ordered_json::parse(tools.dump());
    in.add_generation_prompt = add_generation_prompt;
    in.extra_context = nlohmann::ordered_json::object();
    in.extra_context["enable_thinking"] = enable_thinking;
    if (!reasoning_effort.empty()) in.extra_context["reasoning_effort"] = reasoning_effort;
    minja::chat_template_options o;
    o.apply_polyfills = false;
    return t_.apply(in, o);
  }

 private:
  minja::chat_template t_;
};
