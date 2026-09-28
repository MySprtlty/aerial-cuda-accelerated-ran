"""
runner: the RAN side of one sweep run, behind one interface so the same
harness can drive cubb_gpu_test_bench now and cuphycontroller later.

    r = get_runner(spec["runner"])          # "testbench" | "fullstack"
    files = r.prepare(spec, run_dir)        # writes the inputs (bench yaml, launch pattern, ...)
    cmd   = r.command(spec, run_dir)        # shell command executed inside the Aerial container
    r.labels                                # which label source collect.py must parse

Paths: run_dir is given as the host path; the container sees the same tree
under CONTAINER_ROOT. Everything the runner writes stays inside run_dir.
"""
import os

import yaml

HOST_ROOT = "/home/troy/build/aerial-cuda-accelerated-ran"
CONTAINER_ROOT = "/opt/nvidia/cuBB"
SWEEP_TV_DIR = CONTAINER_ROOT + "/testVectors/GPU_test_input"
MANIFEST = HOST_ROOT + "/testVectors/sweep_tv_manifest.yaml"
BENCH = CONTAINER_ROOT + "/build.aarch64/testBenches/cubb_gpu_test_bench/cubb_gpu_test_bench"

UL_CHANNELS = ("PUSCH", "PUCCH", "PRACH", "SRS")
DL_CHANNELS = ("PDSCH", "PDCCH", "SSB", "CSIRS")
# --M order of cubb_gpu_test_bench: [PRACH --P][PDCCH --Q][PUCCH --X] PDSCH PUSCH [SSB][SRS --Z]
SM_ORDER = ("PRACH", "PDCCH", "PUCCH", "PDSCH", "PUSCH", "SSB", "SRS")
CTX_FLAG = {"PRACH": "--P", "PDCCH": "--Q", "PUCCH": "--X", "SSB": "--B", "SRS": "--Z"}


def to_container(host_path):
    return host_path.replace(HOST_ROOT, CONTAINER_ROOT, 1)


_manifest = None


def manifest():
    global _manifest
    if _manifest is None:
        d = yaml.safe_load(open(MANIFEST))
        _manifest = {int(t["id"]): t for t in d["tvs"]}
    return _manifest


LIBRARY_TV_DIR = CONTAINER_ROOT + "/testVectors"       # NVIDIA's shipped cuPHY TVs (tv_index.csv), referenced by file name


def tv_file(tv):
    """cuPHY h5 file name of a TV: an int = sweep TV id (from the manifest, None if skipped);
    a str = a file name from the shipped library (testVectors/), used as is."""
    if isinstance(tv, str) and not tv.isdigit():
        return tv if os.path.exists(os.path.join(HOST_ROOT, "testVectors", tv)) else None
    t = manifest().get(int(tv))
    if t is None or t.get("status") != "ok":
        return None
    files = t.get("actual", {}).get("cuphy_files") or []
    return files[0] if files else None


def tv_path(tv):
    """Container path of a TV's cuPHY file (sweep dir for ids, library dir for names)."""
    f = tv_file(tv)
    if f is None:
        return None
    return (LIBRARY_TV_DIR if isinstance(tv, str) and not tv.isdigit() else SWEEP_TV_DIR) + "/" + f


class Runner:
    name = "base"
    labels = ()

    def prepare(self, spec, run_dir):
        raise NotImplementedError

    def command(self, spec, run_dir):
        raise NotImplementedError

    def snapshot(self, spec):
        """Static configuration worth recording with the run."""
        return {}


class TestBenchRunner(Runner):
    """cubb_gpu_test_bench, DDDSUUDDDD pattern (-u 5): 8 slots per pattern,
    slots 0-1 uplink (PUSCH, PUCCH; PRACH and SRS in slot 0 only), 2-7 downlink."""
    name = "testbench"
    labels = ("bench_stdout", "nsys_kernels")

    def prepare(self, spec, run_dir):
        p = spec["point"]
        cells = int(p.get("cells", 4))
        chans = {k: v for k, v in p.get("channels", {}).items() if v}
        light = {k: v for k, v in (p.get("light_channels") or {}).items() if v}
        skew = p.get("cell_skew", "uniform")
        files = {}
        for ch, tv in chans.items():
            f = tv_path(tv)
            if f is None:
                raise RuntimeError("TV %s for %s is missing/skipped in the manifest" % (tv, ch))
            files[ch] = f
        light_files = {ch: tv_path(tv) for ch, tv in light.items() if tv_path(tv)}

        def per_cell(ch):
            heavy = files[ch]
            if skew == "cell0" and ch in light_files:
                return [heavy] + [light_files[ch]] * (cells - 1)
            return [heavy] * cells

        slots = []
        for s in range(8):
            entry = {}
            if s < 2:
                for ch in ("PUSCH", "PUCCH"):
                    if ch in files:
                        entry[ch] = per_cell(ch)
                if s == 0:
                    for ch in ("PRACH", "SRS"):
                        if ch in files:
                            entry[ch] = per_cell(ch)
            else:
                for ch in ("PDSCH", "PDCCH", "SSB", "CSIRS"):
                    if ch in files:
                        entry[ch] = per_cell(ch)
            slots.append(entry)
        y = {"cells": cells, "slots": slots}
        path = os.path.join(run_dir, "bench.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(y, fh, sort_keys=False)
        return {"bench_yaml": path, "channels": sorted(files), "files": files, "light_files": light_files}

    def command(self, spec, run_dir):
        p = spec["point"]
        b = spec.get("bench", {})
        chans = [ch for ch, v in p.get("channels", {}).items() if v]
        sm = dict(b.get("sm", {}))
        flags = ["--U", "--D"]
        sm_list = []
        for ch in SM_ORDER:
            if ch in ("PDSCH", "PUSCH"):
                sm_list.append(str(sm.get(ch, 100)))
                continue
            if ch in chans:
                flags.append(CTX_FLAG[ch])
                sm_list.append(str(sm.get(ch, 8)))
        if b.get("group", True):
            flags += ["--G", "--g"]
        iters = int(spec["duration_s"] * 1000.0 / b.get("pattern_ms", 5.0))
        cmd = [BENCH, "-i", to_container(os.path.join(run_dir, "bench.yaml")), "-r", str(iters), "-w", str(b.get("w_us", 1000)),
               "-u", "5", "-d", "0", "-m", str(b.get("mode", 0)), "-c", str(b.get("cpu", 44))] + flags + ["--M", ",".join(sm_list)]
        return " ".join(cmd)

    def snapshot(self, spec):
        p = spec["point"]
        out = {"runner": self.name, "bench": BENCH, "sm_requested": spec.get("bench", {}).get("sm", {}), "tvs": {}}
        for ch, tv in p.get("channels", {}).items():
            if not tv:
                continue
            lib = isinstance(tv, str) and not tv.isdigit()
            t = {} if lib else manifest().get(int(tv), {})
            out["tvs"][ch] = {"id": tv if lib else int(tv), "name": t.get("name", "library"), "requested": t.get("requested"), "file": tv_file(tv),
                              "note": t.get("note"), "check": t.get("check")}
        return out


class FullStackRunner(Runner):
    """cuphycontroller + testMAC + RU emulator (phase4 scripts). Not implemented yet.
    Plan: prepare() writes testVectors/multi-cell/launch_pattern_nrSim_9<TC>.yaml from the FAPI
    TVs of the point and runs cubb_scripts/autoconfig; command() is run1/run2/run3 with
    --timeout; labels come from the [PHYDRV] "... times ===>" NVLOGI lines (nvlog PHYDRV at
    INFO) and the dApp ring (SLOT_END), parsed by collect.py."""
    name = "fullstack"
    labels = ("phydrv_log", "ring_slot_end")

    def prepare(self, spec, run_dir):
        raise NotImplementedError("FullStackRunner: stage 2")

    def command(self, spec, run_dir):
        raise NotImplementedError("FullStackRunner: stage 2")


def get_runner(name):
    return {"testbench": TestBenchRunner, "fullstack": FullStackRunner}[name]()
