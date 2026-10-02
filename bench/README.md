# Oracle parity & benchmark harness

Compares this ggml implementation against the official ONNX export
(`xDoc/model/PP-DocLayoutV3.onnx`, Paddle2ONNX, the model the xDoc app runs)
over real document pages.

## Full-corpus oracle (`oracle_all.py`)

Streams every local PDF through render -> ONNX oracle -> ggml batch ->
per-page verdict, with byte-level (md5) and page-level (thumbnail) dedup,
32-page chunks, and resume via `results.jsonl`:

```sh
python oracle_all.py [--exe ../build-verify/Release/ppdoclayout.exe] [--pdf SUBSTR]
```

Last full run (2026-10-03, RTX 5080, ONNX Runtime 1.30 CPU as oracle):
**4261 unique pages from 37 PDFs — 4261 pass, 0 fail.** Every confident
detection (score >= 0.30) on every page: identical class, identical relative
reading order, boxes within 0.14 px (sub-pixel), masks IoU >= 0.99.
Observed cross-backend FP32 tail, confined to single hard queries on
dense-formula book pages (《特殊函数概论》/《积分方程》): score differs by up to
2.6e-3 on 1 of ~50 dets while its class/box/mask stay identical. Verdicts:
- `bgswap` (informational): absolute order RANKS of background detections
  shift; the relative order of real content never does (`relOrd` gate).
- thresholds used: matching pool >= 0.28 (straddler band), box gate 0.25 px,
  score gate 5e-3, mask IoU > 0.95, relative order exact.

## Small fixed suite (`prep_pdfs.py` / `oracle_onnx.py` / `run_ggml.py`)

- `corpus/` — 6 PDFs (EN papers, ZH slides, formula-heavy proof, ZH report)
- `prep_pdfs.py` — renders 2 pages per PDF into `inputs/*.bin`: 3x800x800 CHW
  float32 /255 (official preprocessing, mean 0 std 1); the same tensor feeds
  both runtimes.
- `oracle_onnx.py` — onnxruntime CPU with `im_shape`/`scale_factor` exactly
  like xDoc's `run_doclayout`; saves `oracle/<tag>.npz`.
- `run_ggml.py` — CLI over every input + parity table.

```sh
python prep_pdfs.py && python oracle_onnx.py
python run_ggml.py --backend cuda [--bench 10]
```
