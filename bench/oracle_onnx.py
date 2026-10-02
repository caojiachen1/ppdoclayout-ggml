#!/usr/bin/env python
"""Run the ONNX oracle (same model xDoc uses) over all prepared inputs.

Feeds im_shape=[[800,800]], scale_factor=[[800/ori_h, 800/ori_w]] exactly like
xDoc's run_doclayout. Saves raw fetches to oracle/<tag>.npz:
  dets  (300,7) float32 [cls, score, x1, y1, x2, y2, order]
  num   ()       int32
  masks (300,200,200) int8-ish
plus per-input wall latency.
"""
import json
import os
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = r"D:/Codebase/xDoc/model/PP-DocLayoutV3.onnx"
IN_DIR = os.path.join(HERE, "inputs")
OUT_DIR = os.path.join(HERE, "oracle")
IMG = 800
NQ = 300

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count()
    sess = ort.InferenceSession(ONNX_PATH, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    print("onnx inputs:", names)

    meta = json.load(open(os.path.join(IN_DIR, "meta.json"), encoding="utf-8"))
    for m in meta:
        x = np.fromfile(m["bin"], dtype=np.float32).reshape(3, IMG, IMG)[None]
        im_shape = np.array([[IMG, IMG]], dtype=np.float32)
        scale = np.array([[IMG / m["ori_h"], IMG / m["ori_w"]]], dtype=np.float32)
        t0 = time.perf_counter()
        o0, o1, o2 = sess.run(None, {names[0]: im_shape, names[1]: x, names[2]: scale})
        dt = (time.perf_counter() - t0) * 1000
        np.savez_compressed(
            os.path.join(OUT_DIR, m["tag"] + ".npz"),
            dets=o0.astype(np.float32), num=o1, masks=(o2 != 0).astype(np.int8),
            ms=dt,
        )
        print(f"  {m['tag']}: {dt:.0f} ms, top5:", [(int(d[0]), round(float(d[1]), 3)) for d in o0[:5]])

if __name__ == "__main__":
    main()
