#!/usr/bin/env python3
"""
sweep: expand matrix.yaml into a run queue and execute it with run_one.py.

    python3 sweep.py --profile pilot [--out runs/<manifest_id>] [--dry-run] [--only SUBSTR] [--skip-tenant yolo26x]

Every run gets runs/<manifest_id>/<run_id>/ (spec.yaml, raw logs, tables, summary.json).
runs/<manifest_id>/manifest.yaml lists the queue with the status of each run and is
rewritten after every run, so re-running the same --out resumes: runs whose
status.yaml says ok are skipped, failed ones are retried. TVs that the sweep
manifest marks skipped make the run "skipped" with the reason. Order: tenant none
first, then yolo26n, then yolo26x (so the isolation baseline exists first).
"""
import argparse
import os
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TENANT_ORDER = ["none", "yolo26n", "yolo26x"]


def grid_points(m):
    pts = {}
    g = m["grids"]["mcs_snr"]
    for mcs in range(g["mcs"][0], g["mcs"][1] + 1):
        for i, snr in enumerate(g["snr_db"]):
            pts["grid_mcs%02d_snr%02d" % (mcs, snr)] = {"PUSCH": 66000 + 10 * mcs + i, "_grid": "mcs_snr", "_mcs": mcs, "_snr": snr}
    g = m["grids"]["cb_count"]
    for ci, cb in enumerate(g["cb"]):
        for d, name in g["dist"].items():
            pts["grid_cb%02d_%s" % (cb, name)] = {"PUSCH": 67000 + 10 * ci + int(d), "_grid": "cb_count", "_cb": cb, "_dist": name}
    return pts


def expand(m, prof, only=None, skip_tenants=()):
    c = m["common"]
    p = m["profiles"][prof]
    if p["points"] == "all":
        points = dict(m["points"])
    elif p["points"] == "grids":
        points = grid_points(m)
    elif p["points"] == "channel_points":      # every point that only swaps channel TVs (no cells/skew points), plus the default
        points = {"default": {}}
        points.update({k: v for k, v in m["points"].items() if k != "default" and not any(str(kk).startswith("_") for kk in v)})
    else:
        points = {k: m["points"][k] for k in p["points"]}
    combos = []
    for t in p["tenants"]:
        if t in skip_tenants:
            continue
        if t == "none":
            for rep in range(int(p.get("repeat_none", 1))):
                combos.append((t, 100, 1, 0, rep))
        else:
            for cap in p["caps"]:
                for b in p["batches"]:
                    for pr in p["priorities"]:
                        combos.append((t, cap, b, pr, 0))
    extra = [(e["tenant"], e["cap"], e["batch"], e["priority"], 0, e.get("point", "default")) for e in p.get("extra", [])]
    runs = []
    for tenant, cap, batch, prio, rep in combos:
        for pname, pdef in points.items():
            runs.append(make_run(m, prof, pname, pdef, tenant, cap, batch, prio, rep))
    for tenant, cap, batch, prio, rep, pname in extra:
        pdef = points.get(pname) or m["points"][pname]
        runs.append(make_run(m, prof, pname, pdef, tenant, cap, batch, prio, rep))
    if p.get("tenant_alone"):
        for tenant, cap, batch, prio, rep in combos:
            if tenant == "none":
                continue
            ra = make_run(m, prof, "default", {}, tenant, cap, batch, prio, rep)
            ra["run_id"] = "tenant_alone__" + ra["run_id"].split("__", 1)[1]
            ra["runner"] = "none"
            ra["point_name"] = "tenant_alone"
            ra["duration_s"] = int(p.get("tenant_alone_s", 60))
            ra["nsys_window_s"] = 0
            ra["_order"] = (-1,) + tuple(ra["_order"][1:])
            runs.append(ra)
    runs.sort(key=lambda r: (TENANT_ORDER.index(r["tenant"]["name"]), r["_order"]))
    if only:
        runs = [r for r in runs if only in r["run_id"]]
    return runs


def make_run(m, prof, pname, pdef, tenant, cap, batch, prio, rep):
    c = m["common"]
    p = m["profiles"][prof]
    chans = dict(c["default_channels"])
    meta = {}
    for k, v in pdef.items():
        if k.startswith("_"):
            meta[k] = v
        else:
            chans[k] = v
    cells = int(meta.get("_cells", p.get("cells", c["cells"])))      # point > profile > common
    skew = meta.get("_cell_skew", c.get("cell_skew", "uniform"))
    tdef = dict(m["tenants"][tenant])
    ten = {"name": tenant}
    if tenant != "none":
        tmpl = p.get("engine_template") or tdef["engine_template"]      # a profile may name its own engines (MIG)
        ten.update(engine=tmpl.format(tenant=tenant, batch=batch, cap=cap), batch=batch, cap_pct=cap, priority=prio)
    bench = dict(c["bench"])
    for k, v in (p.get("bench") or {}).items():                      # profile overrides, e.g. the SM list for a 64-SM MIG instance
        bench[k] = dict(bench.get(k, {}), **v) if isinstance(v, dict) else v
    run_id = "%s__%s" % (pname, "none%d" % rep if tenant == "none" else "%s_cap%03d_b%d_p%d" % (tenant, cap, batch, prio))
    spec = {"run_id": run_id, "profile": prof, "point_name": pname, "runner": c["runner"],
            "point": {"cells": cells, "cell_skew": skew, "channels": chans, "light_channels": c.get("light_channels", {}), "meta": meta},
            "tenant": ten, "tenant_cores": c.get("tenant_cores", "48-51"),
            "duration_s": p.get("duration_s", c["duration_s"]), "warmup_s": p.get("warmup_s", c["warmup_s"]),
            "nsys_window_s": p.get("nsys_window_s", c["nsys_window_s"]), "bench": bench, "mig": p.get("mig"), "_order": (list(m["points"]).index(pname) if pname in m["points"] else 1000, cap, batch, prio, rep)}
    # TV availability
    missing = [(ch, tv) for ch, tv in chans.items() if tv and R.tv_file(tv) is None]
    if missing:
        spec["skip_reason"] = "TV not generated/skipped in sweep_tv_manifest: %s" % ", ".join("%s=%s" % mv for mv in missing)
    if tenant != "none" and not os.path.exists(os.path.join(HERE, "..", "models", ten["engine"])):
        spec["skip_reason"] = "engine missing: models/%s (run models/build_engines.sh)" % ten["engine"]
    return spec


def load_status(run_dir):
    p = os.path.join(run_dir, "status.yaml")
    if os.path.exists(p):
        try:
            return yaml.safe_load(open(p)).get("state", "?")
        except Exception:
            return "?"
    return "pending"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=os.path.join(HERE, "matrix.yaml"))
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-tenant", action="append", default=[])
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--points-file", action="append", default=[], help="yaml with extra `points:` (e.g. runs/c8_scan/minmax_points.yaml from pick_minmax.py)")
    a = ap.parse_args()
    m = yaml.safe_load(open(a.matrix))
    for pf in a.points_file:
        m["points"].update(yaml.safe_load(open(pf)).get("points", {}))
    runs = expand(m, a.profile, a.only or None, a.skip_tenant)
    mid = a.out or os.path.join(HERE, "runs", "%s_%s" % (a.profile, time.strftime("%Y%m%d_%H%M")))
    os.makedirs(mid, exist_ok=True)
    per_run = m["profiles"][a.profile].get("duration_s", m["common"]["duration_s"]) + m["profiles"][a.profile].get("warmup_s", m["common"]["warmup_s"]) + 90
    n_ok = sum(1 for r in runs if load_status(os.path.join(mid, r["run_id"])) == "ok")
    n_skip = sum(1 for r in runs if r.get("skip_reason"))
    print("profile %s: %d runs (%d already ok, %d skipped for missing TVs), ~%.1f h remaining at %.0f s/run -> %s" % (
        a.profile, len(runs), n_ok, n_skip, (len(runs) - n_ok - n_skip) * per_run / 3600.0, per_run, mid))
    manifest = {"profile": a.profile, "matrix": a.matrix, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": []}

    def write_manifest():
        manifest["runs"] = [{"run_id": r["run_id"], "point": r["point_name"], "tenant": r["tenant"], "cells": r["point"]["cells"],
                             "channels": r["point"]["channels"], "state": (load_status(os.path.join(mid, r["run_id"])) if load_status(os.path.join(mid, r["run_id"])) == "ok" else ("skipped" if r.get("skip_reason") else load_status(os.path.join(mid, r["run_id"])))),
                             "skip_reason": r.get("skip_reason", "")} for r in runs]
        yaml.safe_dump(manifest, open(os.path.join(mid, "manifest.yaml"), "w"), sort_keys=False)

    write_manifest()
    if a.dry_run:
        for r in runs:
            print("  %-45s cells=%d channels=%s %s" % (r["run_id"], r["point"]["cells"], r["point"]["channels"], "SKIP: " + r["skip_reason"] if r.get("skip_reason") else ""))
        return 0
    done = 0
    for r in runs:
        rd = os.path.join(mid, r["run_id"])
        if load_status(rd) == "ok":       # never overwrite a finished run (its inputs may have changed since)
            continue
        if r.get("skip_reason"):
            os.makedirs(rd, exist_ok=True)
            yaml.safe_dump({"run_id": r["run_id"], "state": "skipped", "reason": r["skip_reason"]}, open(os.path.join(rd, "status.yaml"), "w"))
            continue
        if a.max_runs and done >= a.max_runs:
            break
        os.makedirs(rd, exist_ok=True)
        spec = {k: v for k, v in r.items() if not k.startswith("_")}
        yaml.safe_dump(spec, open(os.path.join(rd, "spec.yaml"), "w"), sort_keys=False)
        t0 = time.time()
        print("=== [%d/%d] %s" % (done + 1, len(runs) - n_ok - n_skip, r["run_id"]), flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "run_one.py"), "--spec", os.path.join(rd, "spec.yaml"), "--out", rd])
        print("    -> %s in %.0f s" % (load_status(rd), time.time() - t0), flush=True)
        done += 1
        write_manifest()
    write_manifest()
    states = [load_status(os.path.join(mid, r["run_id"])) for r in runs]
    print("done: %s" % {s: states.count(s) for s in set(states)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
