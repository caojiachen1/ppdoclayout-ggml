#!/usr/bin/env python
"""Run ppdoclayout.exe over all inputs, then diff against the ONNX oracle.

Usage: python run_ggml.py [--backend cuda|cpu] [--exe path] [--bench N]
Writes ggml_out/<tag>/out{0,1,2}.bin and prints a parity report.
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "inputs")
ORACLE = os.path.join(HERE, "oracle")
GGML_OUT = os.path.join(HERE, "ggml_out")

def run_one(exe, model, m, backend, bench, out_root):
    od = os.path.join(out_root, m["tag"])
    os.makedirs(od, exist_ok=True)
    cmd = [exe, "-m", model, "-i", m["bin"], "--ori-h", str(m["ori_h"]),
           "--ori-w", str(m["ori_w"]), "-o", od, "--backend", backend]
    if bench:
        cmd += ["--bench", str(bench)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    wall = (time.perf_counter() - t0) * 1000
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f"ggml failed on {m['tag']}")
    bench_line = [l for l in r.stdout.splitlines() if l.startswith("bench_ms")]
    inf_line = [l for l in r.stderr.splitlines() if l.startswith("inference:")]
    return od, wall, (bench_line[0] if bench_line else (inf_line[0] if inf_line else "?"))

CONF_T = 0.30   # gate tier: every det above this must match
POOL_T = 0.28   # matching pool extends below the gate (threshold straddlers)
MATCH_PX = 0.25 # box-match radius / box gate
SCORE_TOL = 5e-3

def compare(tag):
    o = np.load(os.path.join(ORACLE, tag + ".npz"))
    gd = np.fromfile(os.path.join(GGML_OUT, tag, "out0.bin"), dtype=np.float32).reshape(300, 7)
    gm = np.fromfile(os.path.join(GGML_OUT, tag, "out2.bin"), dtype=np.int32).reshape(300, 200, 200)
    od = o["dets"]
    assert od.shape == (300, 7), od.shape
    label_eq = (od[:, 0].astype(int) == gd[:, 0].astype(int))
    score_d = np.abs(od[:, 1] - gd[:, 1])
    box_d = np.abs(od[:, 2:6] - gd[:, 2:6]).max(axis=1)
    order_eq = od[:, 6].astype(int) == gd[:, 6].astype(int)
    om = o["masks"] != 0
    gmb = gm != 0
    def pair_iou(r, j):
        inter = int((om[r] & gmb[j]).sum()); union = int((om[r] | gmb[j]).sum())
        return inter / union if union else 1.0
    keep = od[:, 1] >= CONF_T
    pool = np.where(od[:, 1] >= POOL_T)[0]
    # selected-set equality ignoring rank: greedy nearest-box match, same class
    def set_sym_diff(a, b):
        used = np.zeros(len(b), bool)
        miss = 0
        for r in a:
            cand = np.where(~used & (b[:, 0].astype(int) == int(r[0])))[0]
            if len(cand) == 0:
                miss += 1
                continue
            d = np.abs(b[cand, 2:6] - r[2:6]).max(axis=1)
            j = cand[np.argmin(d)]
            if d.min() > MATCH_PX:
                miss += 1
            else:
                used[j] = True
        return miss
    return {
        "label_mismatch": int((~label_eq).sum()),
        "label_mismatch_conf": int((~label_eq)[keep].sum()),
        "score_maxdiff": float(score_d.max()),
        "score_maxdiff_conf": float(score_d[keep].max()) if keep.any() else 0.0,
        "box_maxdiff_px": float(box_d[keep].max()) if keep.any() else 0.0,
        "box_maxdiff_all": float(box_d.max()),
        "order_mismatch": int((~order_eq).sum()),
        "order_mismatch_conf": int((~order_eq)[keep].sum()),
        "mask_iou_min": float(min(pair_iou(int(r), int(np.argmin(np.abs(gd[:, 2:6] - od[r, 2:6]).max(axis=1)))) for r in np.where(keep)[0])) if keep.any() else 1.0,
        "set_sym_diff": int(set_sym_diff(od[pool], gd) + set_sym_diff(gd[gd[:, 1] >= POOL_T], od)),
        "n_conf": int(keep.sum()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--model", default=os.path.join(HERE, "model-f32.gguf"))
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--out-root", default=GGML_OUT)
    ap.add_argument("--only", default=None, help="comma separated tags")
    args = ap.parse_args()

    exe = args.exe or os.path.join(HERE, "..", "build-cuda", "Release", "ppdoclayout.exe")
    exe = os.path.abspath(exe)
    dll_dir = os.path.abspath(os.path.join(os.path.dirname(exe), "..", "bin", "Release"))
    if os.path.isdir(dll_dir):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    meta = json.load(open(os.path.join(IN_DIR, "meta.json"), encoding="utf-8"))
    if args.only:
        want = set(args.only.split(","))
        meta = [m for m in meta if m["tag"] in want]

    rows = []
    for m in meta:
        od, wall, bench = run_one(exe, args.model, m, args.backend, args.bench, args.out_root)
        c = compare(m["tag"])
        rows.append((m["tag"], wall, bench, c))

    print(f"\n{'input':22s} {'wall_ms':>8s} {'bench':>24s} " +
          "labMM(c) scoreMax(c) boxMM(c) ordMM(c) iouMin(c) setSD  bgswap")
    fail = 0
    for tag, wall, bench, c in rows:
        bad = (c["label_mismatch_conf"] > 0 or c["score_maxdiff_conf"] > SCORE_TOL or
               c["box_maxdiff_px"] > MATCH_PX or c["order_mismatch_conf"] > 0 or
               c["mask_iou_min"] < 0.95 or c["set_sym_diff"] > 0)
        fail += bad
        print(f"{tag:22s} {wall:8.0f} {bench:>24s} " +
              f"{c['label_mismatch_conf']:7d} {c['score_maxdiff_conf']:10.2e} " +
              f"{c['box_maxdiff_px']:7.3f} {c['order_mismatch_conf']:8d} " +
              f"{c['mask_iou_min']:10.6f} {c['set_sym_diff']:5d} {c['order_mismatch']:6d}" +
              ("   <== FAIL" if bad else ""))
    print(f"\n{len(rows) - fail}/{len(rows)} PASS "
          f"(conf>={CONF_T}: label/order exact, score<{SCORE_TOL}, box<{MATCH_PX}px, iou>0.95; selected-set equal; bgswap=adjacent background rank swaps)")
    raise SystemExit(1 if fail else 0)

if __name__ == "__main__":
    main()
