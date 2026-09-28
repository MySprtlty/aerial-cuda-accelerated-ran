#!/usr/bin/env python3
"""
run_one: execute ONE sweep combination end to end and leave everything in
runs/<manifest_id>/<run_id>/.

    python3 run_one.py --spec spec.yaml --out runs/<mid>/<run_id> [--no-collect]

Steps (host side, Aerial pieces via docker exec):
  1. snapshot.yaml   GPU state (nvidia-smi), MPS defaults, TV manifest entries, bench command
  2. telemetry       dcgmi dmon (1 s) on the host -> dcgm.csv; nvidia-smi -l 1 -> smi.csv
  3. tenant          yolo_worker.py in the container as an MPS client with cap/priority env,
                     pinned to spec["tenant_cores"]; wait for its ready file, then warm-up seconds
  4. RAN             runner.command() under nsys (CUDA kernel trace for spec["nsys_window_s"] s,
                     --kill=none so the bench keeps running) -> bench.out, nsys_trace.nsys-rep
  5. stop tenant and telemetry, then collect.py (in the container) -> parquet/csv/summary.json
  6. status.yaml     ok | failed with the reason; a failed run never stops the sweep

Spec keys: run_id, runner, point{cells, cell_skew, channels{CH: tv_id}, light_channels},
tenant{name, engine, batch, cap_pct, priority}, duration_s, warmup_s, nsys_window_s, bench{...}.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner as R  # noqa: E402

CONTAINER = "c_aerial_troy"
DCGM_FIELDS = "1002,1003,1004,1005,203,204,155,100,101"   # SM active, SM occupancy, tensor, DRAM active, GPU util, mem util, power, SM clk, mem clk
MODELS_DIR = R.CONTAINER_ROOT + "/cuPHY-CP/dapp_hook/models"
SWEEP_DIR = R.CONTAINER_ROOT + "/cuPHY-CP/dapp_hook/sweep"


# environment of every docker exec; main() points the MPS pipe at the RAN instance's daemon when the spec has a `mig` block
BASE_ENV = {"CUDA_MPS_PIPE_DIRECTORY": "/var", "CUDA_DEVICE_MAX_CONNECTIONS": "8"}


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def dexec(cmd, env=None, user="root", detach=False, stdout=None):
    args = ["docker", "exec"]
    if detach:
        args.append("-d")
    full_env = dict(BASE_ENV)
    full_env.update(env or {})
    args += ["-u", user]
    for k, v in full_env.items():
        if v is not None:
            args += ["-e", "%s=%s" % (k, v)]
    args += [CONTAINER, "bash", "-lc", cmd]
    if detach:
        return subprocess.run(args, capture_output=True, text=True)
    return subprocess.run(args, capture_output=True, text=True, stdout=None if stdout is None else stdout)


def gpu_snapshot():
    q = "name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,power.limit,temperature.gpu,memory.used,compute_mode,mig.mode.current"
    r = sh("nvidia-smi --query-gpu=%s --format=csv,noheader" % q)
    out = {"nvidia_smi": r.stdout.strip()}
    out["devices"] = sh("nvidia-smi -L").stdout.strip().splitlines()
    r = dexec("echo get_default_active_thread_percentage | nvidia-cuda-mps-control; echo get_server_list | nvidia-cuda-mps-control; echo get_default_client_priority | nvidia-cuda-mps-control")
    out["mps"] = r.stdout.split()
    out["mps_daemon_running"] = bool(sh("pgrep -x nvidia-cuda-mps-control").stdout.strip())
    out["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-collect", action="store_true")
    a = ap.parse_args()
    spec = yaml.safe_load(open(a.spec))
    mig = spec.get("mig") or {}
    if mig:     # the bench (and the MPS queries) talk to the MPS daemon that was started inside the RAN MIG instance
        BASE_ENV["CUDA_MPS_PIPE_DIRECTORY"] = mig["ran"]["mps_pipe"]
    run_dir = os.path.abspath(a.out)
    os.makedirs(run_dir, exist_ok=True)
    yaml.safe_dump(spec, open(os.path.join(run_dir, "spec.yaml"), "w"), sort_keys=False)
    status = {"run_id": spec["run_id"], "state": "running", "t_start": time.time()}
    log = open(os.path.join(run_dir, "run_one.log"), "a")

    def note(msg):
        line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    procs = []
    tenant = spec.get("tenant") or {}
    # never start on top of a leftover bench/worker/nsys from an earlier run
    dexec("pkill -9 -x cubb_gpu_test_b; pkill -9 -f 'yolo_worke[r]'; sleep 1; true")
    try:
        if spec.get("runner", "testbench") == "none":      # tenant alone: no RAN workload
            rn, files, ran_cmd = None, {}, ""
            snap = {"gpu": gpu_snapshot(), "runner": {"runner": "none"}, "command": "", "files": {}, "tenant": tenant}
        else:
            rn = R.get_runner(spec.get("runner", "testbench"))
            files = rn.prepare(spec, run_dir)
            ran_cmd = rn.command(spec, run_dir)
            snap = {"gpu": gpu_snapshot(), "runner": rn.snapshot(spec), "command": ran_cmd, "files": files, "tenant": tenant}
        yaml.safe_dump(snap, open(os.path.join(run_dir, "snapshot.yaml"), "w"), sort_keys=False)

        # telemetry (host)
        dcgm = open(os.path.join(run_dir, "dcgm.csv"), "w")
        procs.append(subprocess.Popen(["dcgmi", "dmon", "-e", DCGM_FIELDS, "-d", "1000"], stdout=dcgm, stderr=subprocess.STDOUT))
        smi = open(os.path.join(run_dir, "smi.csv"), "w")
        procs.append(subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm,power.draw,temperature.gpu",
                                       "--format=csv", "-l", "1"], stdout=smi, stderr=subprocess.STDOUT))

        # tenant
        warm = float(spec.get("warmup_s", 30))
        if tenant and tenant.get("name", "none") != "none":
            ready = os.path.join(run_dir, "worker.ready")
            if os.path.exists(ready):
                os.remove(ready)
            env = {}
            cap = int(tenant.get("cap_pct", 100))
            if cap < 100:
                env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(cap)
            if int(tenant.get("priority", 0)):
                env["CUDA_MPS_CLIENT_PRIORITY"] = "1"
            if mig:     # the tenant owns its MIG instance: direct CUDA on that device, no MPS daemon, no cap
                env = {"CUDA_VISIBLE_DEVICES": mig["tenant"]["uuid"], "CUDA_MPS_PIPE_DIRECTORY": "/var/mig_tenant_no_mps"}
            cores = spec.get("tenant_cores", "48-51")
            wcmd = ("cd %s && taskset -c %s python3 %s/yolo_worker.py --engine %s/%s --input %s/bus_640.npy --batch %d "
                    "--log %s/worker.csv --ready %s --seconds %d > %s/worker.out 2>&1" % (
                        R.to_container(run_dir), cores, SWEEP_DIR, MODELS_DIR, tenant["engine"], MODELS_DIR, int(tenant.get("batch", 1)),
                        R.to_container(run_dir), R.to_container(ready), int(warm + spec["duration_s"] + 120), R.to_container(run_dir)))
            note("tenant: " + wcmd)
            dexec(wcmd, env=env, detach=True)
            t0 = time.time()
            wout = os.path.join(run_dir, "worker.out")
            while not os.path.exists(ready):
                if time.time() - t0 > 300:
                    raise RuntimeError("tenant did not become ready in 300 s (see worker.out)")
                if os.path.exists(wout) and ("Traceback" in open(wout, errors="replace").read()):
                    raise RuntimeError("tenant worker crashed while loading (see worker.out)")
                time.sleep(0.5)
            note("tenant ready after %.0f s, warming up %.0f s" % (time.time() - t0, warm))
            time.sleep(warm)
        else:
            note("no tenant")

        # RAN under nsys (or just let the tenant run for duration_s when there is no RAN)
        if rn is None:
            note("tenant alone for %d s" % int(spec["duration_s"]))
            time.sleep(float(spec["duration_s"]))
            raise StopIteration
        win = int(spec.get("nsys_window_s", 60))
        nsys = ("nsys profile -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --duration=%d --kill=none "
                "-o %s/nsys_trace --force-overwrite true " % (win, R.to_container(run_dir))) if win > 0 else ""
        if nsys and mig.get("nsys_directives"):
            # with MIG on, nsys counts the MIG devices as GPUs, fails its Grace (TH500) platform check and exits with
            # "Unexpected TH500 socket configuration"; the directive states the real configuration (1 CPU + 1 GPU)
            nsys = "NSYS_CONFIG_DIRECTIVES='%s' %s" % (mig["nsys_directives"], nsys)
        # nsys returns when its --duration window ends (--kill=none leaves the bench running), so wait for the
        # bench process itself before declaring the run finished; a leftover bench would contaminate the next run.
        cdir = R.to_container(run_dir)
        full = ("rm -f /tmp/cubb_gpu_test_bench.log; cd %s && %s%s > %s/bench.out 2>&1; echo $? > %s/nsys.exit; "
                "while pgrep -x cubb_gpu_test_b >/dev/null; do sleep 1; done; echo done > %s/bench.exit" % (
                    cdir, nsys, ran_cmd, cdir, cdir, cdir))
        note("ran: " + ran_cmd)
        t_ran = time.time()
        r = dexec(full)
        status["ran_wall_s"] = round(time.time() - t_ran, 1)
        out = open(os.path.join(run_dir, "bench.out"), errors="replace").read() if os.path.exists(os.path.join(run_dir, "bench.out")) else ""
        ok = ("Slot pattern #" in out) and ("run time" in out)
        note("ran finished in %.0f s, timing lines=%s" % (status["ran_wall_s"], ok))
        if not ok:
            raise RuntimeError("RAN command produced no timing output (see bench.out)")
    except StopIteration:
        pass
    except Exception as e:
        status["state"] = "failed"
        status["error"] = repr(e)
        note("FAILED: %r" % (e,))
    finally:
        if tenant and tenant.get("name", "none") != "none":
            dexec("pkill -TERM -f 'yolo_worke[r].py --engine' || true")
            time.sleep(2)
        for p in procs:
            try:
                p.send_signal(signal.SIGINT)
                p.wait(timeout=5)
            except Exception:
                p.kill()

    if status["state"] == "running":
        status["state"] = "ok"
        if not a.no_collect:
            t_c = time.time()
            r = dexec("cd %s && python3 %s/collect.py --run %s > %s/collect.log 2>&1; echo $?" % (
                R.to_container(run_dir), SWEEP_DIR, R.to_container(run_dir), R.to_container(run_dir)))
            status["collect_exit"] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
            status["collect_wall_s"] = round(time.time() - t_c, 1)
            note("collect exit=%s in %.0f s" % (status["collect_exit"], status["collect_wall_s"]))
    status["t_end"] = time.time()
    status["wall_s"] = round(status["t_end"] - status["t_start"], 1)
    yaml.safe_dump(status, open(os.path.join(run_dir, "status.yaml"), "w"), sort_keys=False)
    log.close()
    return 0 if status["state"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
