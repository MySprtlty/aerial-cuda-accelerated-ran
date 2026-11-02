#!/usr/bin/env python3
"""seg_split.py --onnx MODEL.onnx --profile profile_sm56.json --out-dir DIR [--bin-ms 0.35] [--max-frontier 6] [--min-ms 0.08]
Cut a TensorRT-profiled ONNX network into sequential segments of about --bin-ms GPU time (measured on the reference partition
of the profile), at positions where few tensors cross the cut, so that every segment can be enqueued on a different CUDA
green-context stream (per-slot SM repartitioning of the YOLO tenant). Steps: shape inference; per-layer times of the trtexec
--exportProfile json attributed to ONNX nodes (fused layer names carry the node names; Myelin/reformat layers are spread over the
nodes between their neighbours); greedy bins along the topological order, cutting only where the frontier (tensors produced
before and consumed after the cut) has <= --max-frontier tensors; onnx.utils.extract_model per segment; manifest.json with the
segments' input/output tensor names, shapes and estimated time. Host venv: /home/troy/.venvs/dapp_yolo/bin/python."""
import argparse, collections, json, os, re
import onnx, onnx.shape_inference, onnx.utils


def norm(s):
    return re.sub(r"_myl\d+_\d+$", "", s).replace(".", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True); ap.add_argument("--profile", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bin-ms", type=float, default=0.35); ap.add_argument("--max-frontier", type=int, default=6); ap.add_argument("--min-ms", type=float, default=0.08)
    ap.add_argument("--allow-int", action="store_true", help="let INT32/INT64/BOOL tensors cross a cut (engines then need int I/O formats, runner dtype-aware)")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    m = onnx.shape_inference.infer_shapes(onnx.load(a.onnx), strict_mode=False); g = m.graph
    inferred = os.path.join(a.out_dir, "model_inferred.onnx"); onnx.save(m, inferred)
    nodes = list(g.node); N = len(nodes)
    by_norm = {}
    for i, n in enumerate(nodes):
        by_norm.setdefault(norm(n.name), i)
    # --- per-node time from the profile
    rows = [r for r in json.load(open(a.profile)) if isinstance(r, dict) and "averageMs" in r]
    t_node = [0.0] * N; pending = 0.0; cursor = 0; unattributed = 0.0
    for r in rows:
        toks = re.findall(r"/model[._]\d+/[^ +(),]*", r["name"])
        idx = sorted({by_norm[norm(t)] for t in toks if norm(t) in by_norm})
        if idx:
            if pending > 0:      # spread the unattributed time over the nodes between the cursor and this layer's first node
                lo, hi = cursor, max(cursor + 1, idx[0] + 1); share = pending / (hi - lo)
                for j in range(lo, hi): t_node[j] += share
                pending = 0.0
            for j in idx: t_node[j] += r["averageMs"] / len(idx)
            cursor = max(cursor, idx[-1] + 1)
        else:
            pending += r["averageMs"]; unattributed += r["averageMs"]
    if pending > 0:
        share = pending / max(1, N - cursor)
        for j in range(cursor, N): t_node[j] += share
    total = sum(t_node)
    # --- frontier per cut position
    cons = collections.defaultdict(list); prod = {}
    for i, n in enumerate(nodes):
        for x in n.input: cons[x].append(i)
        for y in n.output: prod[y] = i
    outs = {o.name for o in g.output}
    def frontier(i):
        return sorted(t for t, p in prod.items() if p <= i and (any(c > i for c in cons.get(t, [])) or (t in outs and i < N - 1)))
    # only float tensors may cross a cut: the engines are built with fp16 I/O, and an INT64 index tensor (TopK, Expand) at a
    # boundary fails the build (seg26n_fine seg24, 2026-10-31)
    etype = {v.name: v.type.tensor_type.elem_type for v in list(g.value_info) + list(g.input) + list(g.output)}
    DT = {1: "fp16", 10: "fp16", 6: "int32", 7: "int64", 9: "bool"}
    def cut_ok(i):
        f = frontier(i)
        return len(f) <= a.max_frontier and all(etype.get(t, 1) in (DT if a.allow_int else (1, 10)) for t in f)
    # --- greedy bins
    cuts = []; acc = 0.0; last = -1
    for i in range(N - 1):
        acc += t_node[i]
        if acc >= a.bin_ms and cut_ok(i):
            cuts.append(i); acc = 0.0; last = i
    if acc < a.min_ms and cuts:      # tiny tail: merge into the previous segment
        cuts.pop()
    bounds = [-1] + cuts + [N - 1]
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(g.value_info) + list(g.input) + list(g.output)}
    ginputs = {i.name for i in g.input}; inits = {t.name for t in g.initializer}
    segs = []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k] + 1, bounds[k + 1]
        # inputs: tensors the segment's nodes consume that were produced before the segment (or are graph inputs);
        # outputs: tensors produced inside the segment that are consumed after it (or are graph outputs). A tensor that merely
        # passes over the segment is neither: it stays in the runner's buffer (a pass-through output would make TensorRT add an
        # identity with a renamed "__" input, which broke the first build).
        inputs, outputs = [], []
        for i in range(lo, hi + 1):
            for x in nodes[i].input:
                if x and x not in inits and (x in ginputs or prod.get(x, -1) < lo) and x not in inputs:
                    inputs.append(x)
            for y in nodes[i].output:
                if y and (any(c > hi for c in cons.get(y, [])) or y in outs) and y not in outputs:
                    outputs.append(y)
        path = os.path.join(a.out_dir, "seg%02d.onnx" % k)
        onnx.utils.extract_model(inferred, path, inputs, outputs, check_model=False)
        est = sum(t_node[lo:hi + 1])
        segs.append({"k": k, "file": os.path.basename(path), "nodes": [lo, hi], "n_nodes": hi - lo + 1, "first": nodes[lo].name, "last": nodes[hi].name,
                     "inputs": inputs, "outputs": outputs, "est_ms_ref": round(est, 3),
                     "shapes": {t: shapes.get(t) for t in inputs + outputs}, "dtypes": {t: DT.get(etype.get(t, 1), "fp16") for t in inputs + outputs}})
        print("seg%02d: nodes %4d..%4d (%3d) est %.3f ms | in %d out %d | last %s" % (k, lo, hi, hi - lo + 1, est, len(inputs), len(outputs), nodes[hi].name))
    man = {"onnx": os.path.abspath(a.onnx), "profile": os.path.abspath(a.profile), "bin_ms": a.bin_ms, "total_ms_ref": round(total, 3),
           "unattributed_ms": round(unattributed, 3), "n_segments": len(segs), "segments": segs}
    json.dump(man, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
    print("%d segments, total %.2f ms (unattributed %.2f spread), manifest %s" % (len(segs), total, unattributed, os.path.join(a.out_dir, "manifest.json")))


if __name__ == "__main__":
    main()
