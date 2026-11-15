#!/usr/bin/env python3
"""patch_yterm.py <tree root> [--check] [--revert]

YOLO term of the in-process SM sizer (2026-11-14). Exact-string edits, each anchor must occur exactly once:
  cuPHY-CP/dapp_hook/include/dapp_hook/dapp_sizer.hpp      knobs, activity mark, second pick with the extra margin, counters
  cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_module.cpp       one start-up log line
  cuPHY-CP/dapp_hook/tenant/dapp_tenant.cu                  activity mark (main loop, free-run inference start), counters in the log line
Works on the hybrid branch (deployed) and on feature/slot_level. --check only reports; --revert swaps new -> old.
"""
import sys

H = "cuPHY-CP/dapp_hook/include/dapp_hook/dapp_sizer.hpp"
N = "cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_module.cpp"
T = "cuPHY-CP/dapp_hook/tenant/dapp_tenant.cu"

EDITS = []


def ed(f, old, new):
    EDITS.append((f, old, new))


# ---- dapp_sizer.hpp ---------------------------------------------------------------------------------------------------
ed(H,
   " * Header-only, no CUDA; the ring record format is untouched.\n */\n",
   " * Header-only, no CUDA; the ring record format is untouched.\n"
   " *\n"
   " * YOLO term (2026-11-14): the fitted model describes PUSCH with the GPU to itself. While the in-process YOLO tenant has an\n"
   " * inference in flight on its own SMs, PUSCH starts later and runs longer (6 cells + 30 req/s: 2.1 % of the completions\n"
   " * past the planning limit and every out-of-order pair, against 0 % / 0 without YOLO). DAPP_SIZER_YOLO_US =\n"
   " * \"slot:us,slot:us,...\" (slot-in-frame 0..19) is an extra completion margin for those slot positions; it is applied\n"
   " * while the tenant has reported a request in hand (sizer_yolo_mark) within the last DAPP_SIZER_YOLO_HOLD_MS (default 1000).\n"
   " * Unset or all zero = off: the plan is the one without this term.\n"
   " */\n")

ed(H,
   "    std::atomic<uint64_t> plans{0}, incomplete{0}, pusch_slots{0}, pdsch_slots{0}, pusch_infeasible{0}, pdsch_infeasible{0};\n",
   "    std::atomic<uint64_t> plans{0}, incomplete{0}, pusch_slots{0}, pdsch_slots{0}, pusch_infeasible{0}, pdsch_infeasible{0};\n"
   "    std::atomic<uint64_t> pusch_yolo_slots{0}, pusch_yolo_up{0}, pusch_yolo_nofit{0};     // YOLO term: PUSCH slots planned with it / moved to a larger profile by it / no profile meets it (largest taken)\n")

ed(H,
   "inline int64_t g_prev_ul_end_abs_ns = 0;     // burst check: completion of the previous U slot's PUSCH (absolute ns)\n",
   "inline int64_t g_prev_ul_end_abs_ns = 0;     // burst check: completion of the previous U slot's PUSCH (absolute ns)\n"
   "// YOLO term (see the file comment): extra PUSCH completion margin per slot-in-frame while the tenant has work\n"
   "inline double  g_sizer_yolo_us[kSzSlotsPerFrame] = {0};\n"
   "inline bool    g_sizer_yolo_on = false;\n"
   "inline int64_t g_sizer_yolo_hold_ns = 1000000000LL;\n"
   "inline std::atomic<int64_t> g_sizer_yolo_seen_ns{0};     // CLOCK_MONOTONIC ns of the tenant's last mark (0 = never)\n"
   "// tenant thread: a YOLO request is in hand (queued, decoding or running)\n"
   "inline void sizer_yolo_mark(int64_t now_mono_ns) noexcept { g_sizer_yolo_seen_ns.store(now_mono_ns, std::memory_order_relaxed); }\n"
   "inline std::string sizer_yolo_desc()\n"
   "{\n"
   "    if (!g_sizer_yolo_on) { return \"off\"; }\n"
   "    std::string s; char b[64];\n"
   "    for (int i = 0; i < kSzSlotsPerFrame; i++) {\n"
   "        if (g_sizer_yolo_us[i] > 0.0) { std::snprintf(b, sizeof(b), \"%s%d:%.0f\", s.empty() ? \"\" : \",\", i, g_sizer_yolo_us[i]); s += b; }\n"
   "    }\n"
   "    std::snprintf(b, sizeof(b), \" us (slot-in-frame:margin), hold %.0f ms\", g_sizer_yolo_hold_ns / 1e6);\n"
   "    return s + b;\n"
   "}\n")

ed(H,
   "// Reads DAPP_SIZER_MODEL (file) and DAPP_SIZER_WAIT_US; returns false (and leaves the sizer disabled) without a model.\n",
   "// Reads DAPP_SIZER_MODEL (file), DAPP_SIZER_WAIT_US and the YOLO term (DAPP_SIZER_YOLO_US, DAPP_SIZER_YOLO_HOLD_MS);\n"
   "// returns false (and leaves the sizer disabled) without a model.\n")

ed(H,
   "    if (const char* w = std::getenv(\"DAPP_SIZER_WAIT_US\")) { g_sizer_wait_max_ns = static_cast<int64_t>(std::atof(w) * 1000.0); }\n"
   "    g_sizer_enabled = true;\n",
   "    if (const char* w = std::getenv(\"DAPP_SIZER_WAIT_US\")) { g_sizer_wait_max_ns = static_cast<int64_t>(std::atof(w) * 1000.0); }\n"
   "    if (const char* y = std::getenv(\"DAPP_SIZER_YOLO_US\")) {     // e.g. 7:120,8:120,17:120,18:120\n"
   "        std::istringstream ys(y); std::string tok;\n"
   "        while (std::getline(ys, tok, ',')) {\n"
   "            const size_t c = tok.find(':');\n"
   "            if (c == std::string::npos || c == 0) { continue; }\n"
   "            char* end = nullptr; const long s = std::strtol(tok.c_str(), &end, 10);\n"
   "            if (end != tok.c_str() + c) { continue; }\n"
   "            const double v = std::atof(tok.c_str() + c + 1);\n"
   "            if (s >= 0 && s < kSzSlotsPerFrame && v > 0.0) { g_sizer_yolo_us[s] = v; g_sizer_yolo_on = true; }\n"
   "        }\n"
   "    }\n"
   "    if (const char* h = std::getenv(\"DAPP_SIZER_YOLO_HOLD_MS\")) { g_sizer_yolo_hold_ns = static_cast<int64_t>(std::atof(h) * 1e6); }\n"
   "    g_sizer_enabled = true;\n")

ed(H,
   "            double t; bool ok;\n"
   "            const int p = sizer_pick(g_sizer_model.pusch, e.pusch, SZ_PUSCH_K, g_sizer_profiles.pusch_sm, g_sizer_profiles.n_pusch, queue_us, slot, t, ok);\n"
   "            pe.ul_prof = static_cast<int8_t>(p);",
   "            double t; bool ok;\n"
   "            int p = sizer_pick(g_sizer_model.pusch, e.pusch, SZ_PUSCH_K, g_sizer_profiles.pusch_sm, g_sizer_profiles.n_pusch, queue_us, slot, t, ok);\n"
   "            // YOLO term: while the tenant has had a request in hand within the hold time, this slot position's extra margin has to fit too\n"
   "            if (g_sizer_yolo_on && ok) {\n"
   "                const double  y    = g_sizer_yolo_us[slot % kSzSlotsPerFrame];\n"
   "                const int64_t seen = g_sizer_yolo_seen_ns.load(std::memory_order_relaxed);\n"
   "                if (y > 0.0 && seen != 0 && t_start - seen <= g_sizer_yolo_hold_ns) {\n"
   "                    double ty; bool oky;\n"
   "                    const int py = sizer_pick(g_sizer_model.pusch, e.pusch, SZ_PUSCH_K, g_sizer_profiles.pusch_sm, g_sizer_profiles.n_pusch, queue_us + y, slot, ty, oky);\n"
   "                    g_sizer_stats.pusch_yolo_slots.fetch_add(1, std::memory_order_relaxed);\n"
   "                    if (py != p) { g_sizer_stats.pusch_yolo_up.fetch_add(1, std::memory_order_relaxed); }\n"
   "                    if (!oky) { g_sizer_stats.pusch_yolo_nofit.fetch_add(1, std::memory_order_relaxed); }\n"
   "                    p = py; t = ty - y;     // t stays the model's prediction at the chosen profile, without the margin (ok: it meets the limit)\n"
   "                }\n"
   "            }\n"
   "            pe.ul_prof = static_cast<int8_t>(p);")

# ---- nv_phy_module.cpp ------------------------------------------------------------------------------------------------
ed(N,
   "                    else\n"
   "                        NVLOGC_FMT(TAG, \"dApp sizer disabled: {}\", serr);\n",
   "                    else\n"
   "                        NVLOGC_FMT(TAG, \"dApp sizer disabled: {}\", serr);\n"
   "                    if (nv::dapp::g_sizer_enabled)\n"
   "                        NVLOGC_FMT(TAG, \"dApp sizer YOLO term: {}\", nv::dapp::sizer_yolo_desc());\n")

# ---- dapp_tenant.cu ---------------------------------------------------------------------------------------------------
ed(T,
   "            const int64_t tnow = now_mono_ns();\n",
   "            const int64_t tnow = now_mono_ns();\n"
   "            if (have_req) { sizer_yolo_mark(tnow); }     // the sizer's YOLO term (dapp_sizer.hpp): a request is in hand\n")

ed(T,
   "        if (k == 0) { pp.t_inf_start = now_mono_ns(); if (pp.tiles_left == tiles_per_request) pp.t_req_start = pp.t_inf_start; }\n",
   "        if (k == 0) { pp.t_inf_start = now_mono_ns(); sizer_yolo_mark(pp.t_inf_start); if (pp.tiles_left == tiles_per_request) pp.t_req_start = pp.t_inf_start; }\n")

ed(T,
   "                    std::fprintf(lf, \" fallback %llu\\n\", static_cast<unsigned long long>(cuphydriver_dapp_fallback_count(1)));\n",
   "                    std::fprintf(lf, \" fallback %llu\", static_cast<unsigned long long>(cuphydriver_dapp_fallback_count(1)));\n"
   "                    if (g_sizer_yolo_on) { std::fprintf(lf, \" | yolo term slots %llu raised %llu nofit %llu\", static_cast<unsigned long long>(g_sizer_stats.pusch_yolo_slots.load()), static_cast<unsigned long long>(g_sizer_stats.pusch_yolo_up.load()), static_cast<unsigned long long>(g_sizer_stats.pusch_yolo_nofit.load())); }\n"
   "                    std::fprintf(lf, \"\\n\");\n")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv; revert = "--revert" in sys.argv
    if len(args) != 1:
        print(__doc__); return 2
    root = args[0].rstrip("/")
    texts = {}; bad = 0; todo = 0
    for i, (f, old, new) in enumerate(EDITS):
        if revert:
            old, new = new, old
        if f not in texts:
            texts[f] = open(root + "/" + f, encoding="utf-8").read()
        s = texts[f]; n_old = s.count(old); n_new = s.count(new)
        # when "new" contains "old" (pure additions) an applied edit shows n_new == 1 and n_old == 1
        applied = (n_new == 1) and (old in new or n_old == 0)
        if applied:
            print("edit %2d %-58s already applied" % (i, f)); continue
        if n_old != 1:
            print("edit %2d %-58s ANCHOR COUNT %d (need 1)" % (i, f, n_old)); bad += 1; continue
        print("edit %2d %-58s %s" % (i, f, "ok (would apply)" if check else "applied")); todo += 1
        texts[f] = s.replace(old, new)
    if bad:
        print("!! %d anchor problem(s); nothing written" % bad); return 1
    if not check:
        for f, s in texts.items():
            open(root + "/" + f, "w", encoding="utf-8").write(s)
    print("%s: %d edit(s) %s" % (root, todo, "pending" if check else "written"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
