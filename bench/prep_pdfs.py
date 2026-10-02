#!/usr/bin/env python
"""Render PDF pages -> deterministic PNG + /255 CHW f32 .bin inputs for both runtimes.

The SAME .bin is fed to onnxruntime (oracle) and ppdoclayout.exe (ggml),
so any rendering ambiguity cancels out. Preprocessing matches the official
PP-DocLayoutV3 spec (resize 800x800, rescale /255, mean=0 std=1) which is
also what the ggml implementation expects.
"""
import json
import os
import sys

import fitz
import numpy as np
from PIL import Image

IMG = 800
ZOOM = 2.0  # deterministic render scale
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
OUT = os.path.join(HERE, "inputs")

def render(pdf_path: str, tag: str, pages):
    doc = fitz.open(pdf_path)
    made = []
    for pno in pages:
        if pno >= len(doc):
            continue
        pm = doc[pno].get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
        img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        w, h = img.size
        png = os.path.join(OUT, f"{tag}_p{pno}.png")
        img.save(png)
        r = img.resize((IMG, IMG), Image.BILINEAR)  # triangle filter, like Rust FilterType::Triangle
        a = np.asarray(r, dtype=np.float32) / 255.0
        chw = a.transpose(2, 0, 1).astype(np.float32).copy()  # 3x800x800 CHW
        binp = os.path.join(OUT, f"{tag}_p{pno}.bin")
        chw.tofile(binp)
        made.append({"tag": f"{tag}_p{pno}", "bin": binp, "png": png,
                     "ori_h": h, "ori_w": w, "pdf": os.path.basename(pdf_path), "page": pno})
    doc.close()
    return made

def main():
    os.makedirs(OUT, exist_ok=True)
    all_inputs = []
    for f in sorted(os.listdir(CORPUS)):
        if not f.endswith(".pdf"):
            continue
        tag = f[:-4]
        all_inputs += render(os.path.join(CORPUS, f), tag, pages=[0, 1])
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(all_inputs, fp, ensure_ascii=False, indent=1)
    print(f"{len(all_inputs)} inputs written to {OUT}")
    for m in all_inputs:
        print(f"  {m['tag']}: {m['ori_w']}x{m['ori_h']} ({m['pdf']} p{m['page']})")

if __name__ == "__main__":
    main()
