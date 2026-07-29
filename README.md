# ppdoclayout-ggml

[PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) document layout analysis inference in pure C++ with [ggml](https://github.com/ggml-org/ggml). CPU and CUDA backends, bit-aligned with the official ONNX export (identical labels, score diff < 1e-6, box diff 0 px, mask IoU ≈ 1.0).

Detects 25 layout element classes (text, title, table, formula, image, ...) with boxes, instance masks and reading order.

## Features

- Single-file C++ implementation of the full RT-DETR-style pipeline: HGNetV2-L backbone, hybrid encoder, deformable-attention decoder, mask head, reading-order head
- C API (`ppdoclayout_c` shared library) + CLI + safe Rust crate
- CPU and CUDA (RTX 20 through RTX 50 / Blackwell) via `ggml_backend_sched`
- FP32 numerical parity with ONNX Runtime, including on GPU (TF32 disabled via a vendored ggml patch, applied automatically at configure time)

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

The first configure initializes the ggml submodule and applies the FP32 parity patch automatically.

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

800×800 input, single image (RTX 4060 / i7, Windows):

| Runtime | Latency |
|---|---|
| onnxruntime (CPU) | ~850 ms |
| ggml (CPU) | ~2100 ms |
| ggml (CUDA) | ~100 ms |

## License

The ggml submodule is MIT licensed. Model weights follow the PaddlePaddle license.
