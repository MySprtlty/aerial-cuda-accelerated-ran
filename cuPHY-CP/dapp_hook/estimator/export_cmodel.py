#!/usr/bin/env python3
"""export_cmodel.py: write the PUSCH / PDSCH part of sm_model_fit.yaml (sm_refit.py output) as the flat "key value" text
file read by the in-process sizer (include/dapp_hook/dapp_sizer.hpp, env DAPP_SIZER_MODEL).

    python3 export_cmodel.py [--model sm_model_fit.yaml] [--out sm_model.txt]

Keys: sm_total, granularity, <CH>.kind (completion|duration), <CH>.s_ref, <CH>.fixed_us, <CH>.budget_us, <CH>.margin_us,
<CH>.complete_by_us, <CH>.start_us.<slot-in-frame>, <CH>.start10_us.<pos>, <CH>.w_ins.<kernel>, <CH>.w_par.<kernel>.
The per-channel budget kind and the matching margin / weights follow sm_model.py (duration fit: w_ins/w_par + margin_us;
completion fit: w_ins_comp/w_par_comp + margin_comp_us when the channel's budget_kind is completion and the fit has them).
"""
import argparse
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "sm_model_fit.yaml"))
    ap.add_argument("--out", default=os.path.join(HERE, "sm_model.txt"))
    a = ap.parse_args()
    y = yaml.safe_load(open(a.model))
    lines = ["# exported from %s" % os.path.abspath(a.model), "sm_total %d" % int(y.get("sm_total", 132)), "granularity %d" % int(y.get("granularity", 8))]
    gkind = str(y.get("budget_kind", "duration"))
    for ch in ("PUSCH", "PDSCH"):
        d = y["channels"][ch]
        kind = str(d.get("budget_kind") or gkind)
        if kind == "completion" and d.get("complete_by_us") is None:
            kind = "duration"
        lines.append("%s.kind %s" % (ch, kind))
        for k in ("s_ref", "fixed_us", "budget_us", "complete_by_us"):
            if d.get(k) is not None:
                lines.append("%s.%s %r" % (ch, k, float(d[k])))
        # weights and margin of the fit that matches the budget kind
        if kind == "completion" and d.get("w_ins_comp"):
            w_ins, w_par, margin = d["w_ins_comp"], d["w_par_comp"], d.get("margin_comp_us", d.get("margin_us", 0.0))
        else:
            w_ins, w_par, margin = d["w_ins"], d["w_par"], d.get("margin_us", 0.0)
        lines.append("%s.margin_us %r" % (ch, float(margin)))
        for k, v in (w_ins or {}).items():
            lines.append("%s.w_ins.%s %r" % (ch, k, float(v)))
        for k, v in (w_par or {}).items():
            lines.append("%s.w_par.%s %r" % (ch, k, float(v)))
        for s, v in sorted((d.get("start_us") or {}).items()):
            s = int(s)
            if s >= 20:
                continue
            lines.append("%s.start_us.%d %r" % (ch, s, float(v)))
            if s < 10 and ("%s.start10_us.%d" % (ch, s)) not in "\n".join(lines):
                lines.append("%s.start10_us.%d %r" % (ch, s, float(v)))
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("wrote %s (%d lines)" % (a.out, len(lines)))
    for ln in lines:
        print("  " + ln)


if __name__ == "__main__":
    main()
