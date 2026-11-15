// sizer_yterm_test.cpp : stand-alone replay of work vectors through the real sizer_plan() (dapp_sizer.hpp), cuphydriver stubbed.
//   sizer_yterm_test <work.txt> <mark: none|fresh|stale>     env: DAPP_SIZER_MODEL, DAPP_SIZER_YOLO_US, DAPP_SIZER_YOLO_HOLD_MS
//   work.txt lines: <slot-in-pattern> chest eq llr ldpc uci ; output lines: <slot-in-pattern> <profile> <sm> <pusch_t> <ok> <set_profile idx>
#include <cstdint>
#include <cstdio>
static int g_last_pusch = -1;
void cuphydriver_dapp_set_profile(uint16_t, uint16_t, uint8_t pusch_idx, uint8_t) { g_last_pusch = pusch_idx; }
int  cuphydriver_dapp_profile_sms(int channel, int* sm, int max_count)
{
    static const int pu[] = {20, 28, 36, 44, 52, 60, 68, 76, 92, 108}; static const int pd[] = {28, 44};
    const int* s = channel == 0 ? pu : pd; const int n = channel == 0 ? 10 : 2;
    for (int i = 0; i < n && i < max_count; i++) { sm[i] = s[i]; }
    return n;
}
uint64_t cuphydriver_dapp_profile_count(int, int) { return 0; }
uint64_t cuphydriver_dapp_fallback_count(int) { return 0; }
#include "dapp_hook/dapp_sizer.hpp"
using namespace nv::dapp;
int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage\n"); return 2; }
    std::string err;
    if (!sizer_init(err)) { std::fprintf(stderr, "sizer_init: %s\n", err.c_str()); return 1; }
#ifndef YTERM_OLD
    std::printf("# yolo term: %s\n", sizer_yolo_desc().c_str());
#endif
    const std::string mark = argv[2];
    std::FILE* f = std::fopen(argv[1], "r"); if (!f) { return 1; }
    int s80; double w[5];
    while (std::fscanf(f, "%d %lf %lf %lf %lf %lf", &s80, &w[0], &w[1], &w[2], &w[3], &w[4]) == 6) {
        const uint16_t sfn = static_cast<uint16_t>(100 + s80 / 20), slot = static_cast<uint16_t>(s80 % 20);
        sizer_expect(sfn, slot, 0);
        SizerSlot& e = sizer_entry(sfn, slot);
        for (int k = 0; k < 5; k++) { e.pusch[k] = w[k]; }
        e.n_pusch = 1; e.done_ul.store(e.expect_ul.load());
#ifndef YTERM_OLD
        if (mark == "fresh") { sizer_yolo_mark(sizer_now_ns()); }
        else if (mark == "stale") { sizer_yolo_mark(sizer_now_ns() - g_sizer_yolo_hold_ns - 5000000LL); }
#endif
        sizer_plan(sfn, slot, 0);
        const SlotPlanEntry& pe = slot_plan_entry(slot_mono_id(sfn, slot));
        std::printf("%d %d %d %.3f %d %d\n", s80, pe.ul_prof, pe.ul_sm, static_cast<double>(pe.pusch_t), pe.pusch_ok, g_last_pusch);
    }
#ifndef YTERM_OLD
    std::printf("# pusch_slots %llu infeasible %llu yolo_slots %llu raised %llu nofit %llu\n",
                static_cast<unsigned long long>(g_sizer_stats.pusch_slots.load()), static_cast<unsigned long long>(g_sizer_stats.pusch_infeasible.load()),
                static_cast<unsigned long long>(g_sizer_stats.pusch_yolo_slots.load()), static_cast<unsigned long long>(g_sizer_stats.pusch_yolo_up.load()),
                static_cast<unsigned long long>(g_sizer_stats.pusch_yolo_nofit.load()));
#else
    std::printf("# pusch_slots %llu infeasible %llu\n", static_cast<unsigned long long>(g_sizer_stats.pusch_slots.load()), static_cast<unsigned long long>(g_sizer_stats.pusch_infeasible.load()));
#endif
    return 0;
}
