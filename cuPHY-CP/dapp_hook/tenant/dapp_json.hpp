/*
 * dapp_json.hpp (2026-11-02): a minimal JSON value + parser + writer for the xApp interface (docs/XAPP_INTERFACE.md).
 * Objects, arrays, strings (with \uXXXX -> UTF-8), numbers, true/false/null. No dependencies.
 */
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace nv { namespace dapp { namespace json {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false; double num = 0.0; std::string str;
    std::vector<Value> arr; std::map<std::string, Value> obj;

    bool is_null() const { return type == Null; }
    const Value* get(const char* k) const { if (type != Object) return nullptr; auto it = obj.find(k); return it == obj.end() ? nullptr : &it->second; }
    std::string s(const char* k, const std::string& d = "") const { const Value* v = get(k); return (v && v->type == String) ? v->str : d; }
    double n(const char* k, double d = 0.0) const { const Value* v = get(k); return (v && v->type == Number) ? v->num : ((v && v->type == Bool) ? (v->b ? 1.0 : 0.0) : d); }
    long long i(const char* k, long long d = 0) const { const Value* v = get(k); return (v && v->type == Number) ? static_cast<long long>(v->num) : d; }
    bool t(const char* k, bool d = false) const { const Value* v = get(k); return (v && v->type == Bool) ? v->b : ((v && v->type == Number) ? v->num != 0.0 : d); }
    bool has_in_array(const char* k, const std::string& item) const { const Value* v = get(k); if (!v || v->type != Array) return false; for (const Value& e : v->arr) { if (e.type == String && e.str == item) return true; } return false; }
};

class Parser {
public:
    explicit Parser(const char* s, size_t n) : p_(s), e_(s + n) {}
    bool parse(Value& out, std::string& err) { ws(); if (!value(out, err)) return false; ws(); if (p_ != e_) { err = "trailing characters"; return false; } return true; }
private:
    const char* p_; const char* e_;
    void ws() { while (p_ < e_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) p_++; }
    bool value(Value& v, std::string& err) {
        if (p_ >= e_) { err = "unexpected end"; return false; }
        switch (*p_) {
            case '{': return object(v, err);
            case '[': return array(v, err);
            case '"': v.type = Value::String; return string(v.str, err);
            case 't': if (e_ - p_ >= 4 && !std::strncmp(p_, "true", 4)) { p_ += 4; v.type = Value::Bool; v.b = true; return true; } break;
            case 'f': if (e_ - p_ >= 5 && !std::strncmp(p_, "false", 5)) { p_ += 5; v.type = Value::Bool; v.b = false; return true; } break;
            case 'n': if (e_ - p_ >= 4 && !std::strncmp(p_, "null", 4)) { p_ += 4; v.type = Value::Null; return true; } break;
            default: break;
        }
        if (*p_ == '-' || (*p_ >= '0' && *p_ <= '9')) { char* end = nullptr; v.num = std::strtod(p_, &end); if (end == p_) { err = "bad number"; return false; } p_ = end; v.type = Value::Number; return true; }
        err = std::string("unexpected character '") + *p_ + "'"; return false;
    }
    bool object(Value& v, std::string& err) {
        v.type = Value::Object; p_++; ws();
        if (p_ < e_ && *p_ == '}') { p_++; return true; }
        for (;;) {
            ws(); if (p_ >= e_ || *p_ != '"') { err = "object key expected"; return false; }
            std::string k; if (!string(k, err)) return false; ws();
            if (p_ >= e_ || *p_ != ':') { err = "':' expected"; return false; } p_++; ws();
            Value item; if (!value(item, err)) return false; v.obj[k] = std::move(item); ws();
            if (p_ < e_ && *p_ == ',') { p_++; continue; }
            if (p_ < e_ && *p_ == '}') { p_++; return true; }
            err = "',' or '}' expected"; return false;
        }
    }
    bool array(Value& v, std::string& err) {
        v.type = Value::Array; p_++; ws();
        if (p_ < e_ && *p_ == ']') { p_++; return true; }
        for (;;) {
            ws(); Value item; if (!value(item, err)) return false; v.arr.push_back(std::move(item)); ws();
            if (p_ < e_ && *p_ == ',') { p_++; continue; }
            if (p_ < e_ && *p_ == ']') { p_++; return true; }
            err = "',' or ']' expected"; return false;
        }
    }
    bool string(std::string& out, std::string& err) {
        p_++; out.clear();
        while (p_ < e_) {
            const char c = *p_++;
            if (c == '"') return true;
            if (c != '\\') { out += c; continue; }
            if (p_ >= e_) break;
            const char x = *p_++;
            switch (x) {
                case '"': out += '"'; break; case '\\': out += '\\'; break; case '/': out += '/'; break;
                case 'b': out += '\b'; break; case 'f': out += '\f'; break; case 'n': out += '\n'; break; case 'r': out += '\r'; break; case 't': out += '\t'; break;
                case 'u': {
                    if (e_ - p_ < 4) { err = "bad \\u escape"; return false; }
                    unsigned cp = static_cast<unsigned>(std::strtoul(std::string(p_, 4).c_str(), nullptr, 16)); p_ += 4;
                    if (cp < 0x80) out += static_cast<char>(cp);
                    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
                    else { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
                    break;
                }
                default: err = "bad escape"; return false;
            }
        }
        err = "unterminated string"; return false;
    }
};

inline bool parse(const std::string& text, Value& out, std::string& err) { Parser p(text.data(), text.size()); return p.parse(out, err); }

inline std::string quote(const std::string& s) {
    std::string o; o.reserve(s.size() + 2); o += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
            default: if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += static_cast<char>(c);
        }
    }
    o += '"'; return o;
}
inline std::string num(double v) { char b[32]; if (v == static_cast<long long>(v) && std::abs(v) < 1e15) std::snprintf(b, sizeof b, "%lld", static_cast<long long>(v)); else std::snprintf(b, sizeof b, "%.3f", v); return b; }

// small incremental writer: Obj o; o.kv("type","x").kv("n",3); o.str()
struct Obj {
    std::string s = "{"; bool first = true;
    Obj& key(const char* k) { if (!first) s += ','; first = false; s += quote(k); s += ':'; return *this; }
    Obj& kv(const char* k, const std::string& v) { key(k); s += quote(v); return *this; }
    Obj& kv(const char* k, const char* v) { key(k); s += quote(v); return *this; }
    Obj& kv(const char* k, double v) { key(k); s += num(v); return *this; }
    Obj& kv(const char* k, int v) { key(k); s += std::to_string(v); return *this; }
    Obj& kv(const char* k, long long v) { key(k); s += std::to_string(v); return *this; }
    Obj& kv(const char* k, unsigned long long v) { key(k); s += std::to_string(v); return *this; }
    Obj& kv(const char* k, bool v) { key(k); s += v ? "true" : "false"; return *this; }
    Obj& raw(const char* k, const std::string& json) { key(k); s += json; return *this; }
    Obj& null(const char* k) { key(k); s += "null"; return *this; }
    std::string str() const { return s + "}"; }
};

}}} // namespace nv::dapp::json
