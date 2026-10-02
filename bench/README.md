# Oracle parity & benchmark harness

Compares this ggml implementation against the official ONNX export
(`xDoc/model/PP-DocLayoutV3.onnx`, Paddle2ONNX, the model the xDoc app runs)
over a corpus of real document pages.

## Files

- `corpus/` — 6 PDFs (EN papers, ZH slides, formula-heavy proof, ZH report)
- `prep_pdfs.py` — renders 2 pages per PDF (PyMuPDF, zoom 2.0) into
  `inputs/*.bin`: 3x800x800 CHW float32, /255 — the exact preprocessing the
  official `preprocessor_config.json` defines (mean 0, std 1) and the same
  tensor is fed to both runtimes.
- `oracle_onnx.py` — runs onnxruntime (CPU) with `im_shape=[[800,800]]`,
  `scale_factor=[[800/h, 800/w]]` exactly like xDoc's `run_doclayout`;
  saves `oracle/<tag>.npz`.
- `run_ggml.py` — runs the CLI on every input and prints the parity table.

## Usage

```sh
python prep_pdfs.py            # inputs/ + meta.json
python oracle_onnx.py          # oracle/*.npz (~0.5 s/page, one-time)
python run_ggml.py --backend cuda [--bench 10] [--only tag1,tag2]
```

## Pass criteria (per page)

- every det with oracle score >= 0.30: label and reading-order rank exactly
  equal, score diff < 1e-4, box diff < 0.05 px, mask IoU > 0.999
- the selected top-300 sets are equal (0.1 px greedy matching)
- `bgswap` counts adjacent swaps among background detections whose scores are
  within float noise of each other (~1e-6). With ~6900 of 7499 adjacent score
  gaps below 1e-5, a handful of such swaps is unavoidable between any two
  GEMM implementations; all confident detections are unaffected.
