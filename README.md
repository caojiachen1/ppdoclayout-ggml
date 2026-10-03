# ppdoclayout-ggml

[PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) document layout analysis inference in pure C++ with [ggml](https://github.com/ggml-org/ggml). CPU and CUDA backends, aligned with the official ONNX export on a multi-PDF oracle (identical labels / reading order / selected top-300, score diff < 3e-6, box diff <= 0.001 px, mask IoU ~ 1.0 for every confident detection — see `bench/`).

Detects 25 layout element classes (text, title, table, formula, image, ...) with boxes, instance masks and reading order.

## Features

- Single-file C++ implementation of the full RT-DETR-style pipeline: HGNetV2-L backbone, hybrid encoder, deformable-attention decoder, mask head, reading-order head
- C API (`ppdoclayout_c` shared library) + CLI + safe Rust crate
- CPU and CUDA (RTX 20 through RTX 50 / Blackwell) via `ggml_backend_sched`
- FP32 numerical parity with ONNX Runtime, including on GPU (TF32 disabled via a vendored ggml patch, applied automatically at configure time)
- CUDA path: whole-graph capture/replay, deformable attention + mask-box ops as
  GPU kernels, and 1x1 / 3x3-stride1 convs decomposed into batched cuBLAS GEMMs
  (weights repacked at load; no im2col materialization, no layout transposes)
- Multi-page oracle harness (`bench/`) against the Paddle2ONNX export that
  [xDoc](https://github.com/caojiachen1/xDoc) ships

## Build

Requires CMake ≥ 3.18, a C++17 compiler, and optionally the CUDA toolkit (≥ 12.8 recommended).

```bat
git clone --recursive https://github.com/caojiachen1/ppdoclayout-ggml
cd ppdoclayout-ggml

rem CPU only
cmake -S . -B build-cpu -DUSE_CUDA=OFF
cmake --build build-cpu --config Release -j

rem CUDA
cmake -S . -B build-cuda -DUSE_CUDA=ON
cmake --build build-cuda --config Release -j
```

The first configure initializes the ggml submodule and applies the
`patches/ggml-ppdl.patch` (FP32 parity + tagged custom-op GPU support)
automatically.

## Get the model

Download the [safetensors weights](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) and convert to GGUF:

```sh
pip install safetensors gguf numpy
python scripts/convert.py   # PP-DocLayoutV3_safetensors/ -> model-f32.gguf
```

## Usage

CLI (input is a raw 3×800×800 CHW float32 RGB tensor, values /255):

```sh
ppdoclayout -m model-f32.gguf -i input.bin --ori-h 2339 --ori-w 1654 -o out --backend cuda
```

C:

```c
ppdl_ctx * ctx = ppdl_init("model-f32.gguf", PPDL_BACKEND_CUDA);
ppdl_result res;
ppdl_infer(ctx, image_chw, ori_h, ori_w, &res);   // res.dets: 300 x [label, score, x1, y1, x2, y2, order]
ppdl_free(ctx);
```

Rust:

```rust
let mut model = ppdoclayout::Model::new("model-f32.gguf", Backend::Cuda)?;
let out = model.infer(&image_chw, 2339, 1654)?;
for det in out.detections().filter(|d| d.score > 0.5) { /* ... */ }
```

Full API reference: [docs/API.en.md](docs/API.en.md) ([中文](docs/API.md)).

## Performance

800×800 input, single image, steady state (`--bench`, warm graph):

| Runtime | Latency |
|---|---|
| onnxruntime 1.30 (CPU, 24T) | 300 – 550 ms |
| ggml (CPU) | ~1600 ms |
| ggml (CUDA, RTX 5080) | **~12 ms** |
| ggml (CUDA, RTX 4060, before the 2026-10 optimization round) | ~100 ms |

CUDA notes: `GGML_CUDA_GRAPHS` is forced on for `USE_CUDA` builds (removes
per-kernel launch overhead of the ~1000-node graph). The 2026-10 perf round
took the RTX 5080 from ~17.6 ms to ~12.1 ms:

- 1x1 convs run as single cuBLAS GEMMs writing CHW directly, with the bias +
  activation fused into one follow-up kernel (tagged custom ops)
- k3 s1 p1 convs run as **Winograd F(4,3)**: input transform → 36-plane
  batched cuBLAS SGEMM (4x fewer FLOPs than the direct convolution) → output
  transform fused with bias + activation (tagged custom ops; matrices solved
  numerically, FP32 tile error ~1e-6, oracle-exact)
- k2 / k3-s2 convs stay on the coalesced F32 im2col + SGEMM path
- numerics stay FP32 throughout (TF32 off), so oracle parity is preserved

The remaining wall time is dominated by FP32 GEMM work (~30 TFLOPS ceiling
with tensor cores unusable under the parity constraint — BF16x9 emulation
measured at 1.0x on this part).

## Correctness

`bench/` holds the oracle harness against the Paddle2ONNX export with
identical input tensors. Full-corpus run (2026-10): **4261 unique pages from
37 local PDFs** (EN/ZH papers, slides, formula-dense math books, engineering
drawings) — every confident detection (score >= 0.3) on every page has an
identical class, identical relative reading order, a box within 0.14 px and
mask IoU >= 0.99. Residual cross-backend FP32 divergence is confined to the
third decimal of individual scores on rare hard queries (max observed 2.6e-3)
and to rank shuffling among background detections; see `bench/README.md`.

2026-10 fix: the reading-order head previously consumed the `dec_norm` output;
it must consume the final decoder hidden state *before* that norm (as in the
ONNX graph). Before this fix reading-order ranks diverged from ONNX on most
pages; they are now exact for all confident detections.

## License

The ggml submodule is MIT licensed. Model weights follow the PaddlePaddle license.
