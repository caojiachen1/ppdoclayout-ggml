# PP-DocLayoutV3 ggml — C/C++ & Rust API Reference

[中文版 / Chinese version](API.md)

A ggml-based inference library for PP-DocLayoutV3 document layout analysis. Output matches the official ONNX export bit-for-bit in practice (identical labels, score error < 1e-6, box error 0 px, mask IoU ≈ 1.0).

## Deliverables

| File | Description |
|---|---|
| `include/ppdoclayout.h` | C API header (usable from C and C++) |
| `ppdoclayout_c.dll` / `.lib` | Shared library (under `build-*/Release/`) |
| `rust/ppdoclayout/` | Rust crate (safe wrapper + example) |
| `model-f32.gguf` | Model weights (generated from safetensors by `scripts/convert.py`) |

Runtime dependencies: `ppdoclayout_c.dll` plus the ggml DLLs next to it (`ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, and `ggml-cuda.dll` for the CUDA build). The executable/process must be able to find them (same directory or `PATH`).

## Building

```bat
rem CPU build
cmake -S . -B build-cpu -DUSE_CUDA=OFF
cmake --build build-cpu --config Release -j

rem CUDA build
cmake -S . -B build-cuda -DUSE_CUDA=ON
cmake --build build-cuda --config Release -j

rem MSVC puts the ggml DLLs in bin\Release; copy them next to the exe
copy /Y build-cuda\bin\Release\*.dll build-cuda\Release\
```

The first configure automatically initializes the ggml submodule and applies the FP32 parity patch (see the GPU support section below).

## Input contract (same for both languages)

The caller is responsible for preprocessing:

1. Resize the original image to **800×800** (tests use `cv2.INTER_CUBIC`; the interpolation method does not affect API correctness, but keep it consistent with whatever you compare against);
2. Divide pixel values by **255**; no mean/std normalization;
3. Layout as **CHW float32** (3×800×800, RGB), 1,920,000 floats total.

Pass the original image size as `ori_h` / `ori_w`; output boxes are automatically mapped back to original-image pixel coordinates.

## Output contract

Every inference returns exactly **300 detection rows**, sorted by score descending:

- `dets`: a `300 × 7` f32 matrix, each row `[label, score, x1, y1, x2, y2, order]`; coordinates are original-image pixels (not clipped, may go out of bounds); `order` is the reading-order rank of this element among all 300 queries (0 = read first).
- `masks`: `300 × 200 × 200` i32 (0/1); mask row i corresponds to dets row i; the mask covers the 800×800 network input (stride 4), so scale it proportionally to map back onto the original image.
- No score-threshold filtering, no NMS (matches the in-graph behavior of the ONNX export). Typical usage keeps rows with `score > 0.5`.

### The 25 class labels

| id | label | id | label | id | label |
|---|---|---|---|---|---|
| 0 | abstract | 9 | footer | 18 | reference |
| 1 | algorithm | 10 | footnote | 19 | reference_content |
| 2 | aside_text | 11 | formula_number | 20 | seal |
| 3 | chart | 12 | header | 21 | table |
| 4 | content | 13 | header | 22 | text |
| 5 | formula | 14 | image | 23 | text |
| 6 | doc_title | 15 | formula | 24 | vision_footnote |
| 7 | figure_title | 16 | number | | |
| 8 | footer | 17 | paragraph_title | | |

(Some ids share the same meaning; this mapping ships with the official checkpoint.)

---

## C / C++ API

### Interface overview

```c
#include "ppdoclayout.h"

ppdl_ctx * ppdl_init   (const char * model_path, int backend);          // backend: PPDL_BACKEND_CPU / PPDL_BACKEND_CUDA
ppdl_ctx * ppdl_init_ex(const char * model_path, int backend, int enable_dumps);
int        ppdl_infer  (ppdl_ctx *, const float * image_chw, int ori_h, int ori_w, ppdl_result *);
int        ppdl_dump   (ppdl_ctx *, const char * dir);                  // requires init_ex(enable_dumps=1); debugging only
void       ppdl_free   (ppdl_ctx *);
```

```c
typedef struct {
    const float   * dets;     // [300*7]
    int32_t         num_dets; // always 300
    const int32_t * masks;    // [300*200*200]
    double          infer_ms; // graph compute time of this call (excludes pre/postprocess)
} ppdl_result;
```

### Semantics and constraints

- **Return values**: `ppdl_init*` returns NULL on failure; `ppdl_infer` / `ppdl_dump` return 0 on success. Error details are printed to stderr.
- **Lifetime**: the pointers in `ppdl_result` reference buffers owned by the ctx and are **invalidated by the next `ppdl_infer` or by `ppdl_free`**. Do not free them; copy the data if you need to keep it.
- **Thread safety**: a single `ppdl_ctx` must not be called concurrently. For multi-threading, create one ctx per thread (each ctx loads its own weights, ~500 MB RAM/VRAM).
- **Reuse**: a ctx can run `infer` repeatedly (the graph is pre-allocated; no extra allocation from the second call on).
- On the CUDA backend, the 3 custom operators (topk / mask_to_box / deformable-attention sampling) automatically fall back to CPU; the scheduler handles the transfers transparently.

### Complete example

```c
#include "ppdoclayout.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    ppdl_ctx * ctx = ppdl_init("model-f32.gguf", PPDL_BACKEND_CUDA);
    if (!ctx) return 1;

    float * img = malloc(sizeof(float) * 3 * PPDL_IMG_SIZE * PPDL_IMG_SIZE);
    // ... fill with CHW RGB /255 data (e.g. loaded from file or resized yourself) ...

    ppdl_result res;
    if (ppdl_infer(ctx, img, /*ori_h=*/2339, /*ori_w=*/1654, &res) != 0) return 1;
    printf("infer %.1f ms\n", res.infer_ms);

    for (int i = 0; i < res.num_dets; i++) {
        const float * d = res.dets + i * PPDL_DET_FIELDS;
        if (d[1] < 0.5f) continue;
        printf("label=%d score=%.3f box=(%.1f,%.1f,%.1f,%.1f) order=%d\n",
               (int) d[0], d[1], d[2], d[3], d[4], d[5], (int) d[6]);
        const int32_t * mask = res.masks + (size_t) i * PPDL_MASK_HW * PPDL_MASK_HW;
        (void) mask; // 200x200, 0/1
    }
    ppdl_free(ctx);
    free(img);
    return 0;
}
```

### CMake integration

```cmake
add_executable(myapp main.c)
target_include_directories(myapp PRIVATE path/to/ppdoclayout-ggml/include)
target_link_libraries(myapp PRIVATE path/to/build-cuda/Release/ppdoclayout_c.lib)
# At run time, place ppdoclayout_c.dll + ggml*.dll next to myapp.exe
```

---

## Rust API

The crate lives in `rust/ppdoclayout` and links against `ppdoclayout_c` via FFI.

### Adding the dependency

```toml
[dependencies]
ppdoclayout = { path = "path/to/ppdoclayout-ggml/rust/ppdoclayout" }
```

Library discovery (`build.rs`): the `PPDL_LIB_DIR` environment variable takes priority; otherwise the in-repo `build-cuda/Release` and `build-cpu/Release` directories are tried automatically. The DLLs must also be on `PATH` at run time:

```bat
set PPDL_LIB_DIR=D:\Codebase\ppdoclayout-ggml\build-cuda\Release
set PATH=%PPDL_LIB_DIR%;%PATH%
cargo run --release --example detect -- model-f32.gguf ref\input.bin cuda 2339 1654
```

### API overview

```rust
pub enum Backend { Cpu, Cuda }

pub struct Model;                      // Send (movable to another thread), not Sync
impl Model {
    pub fn new(path: impl AsRef<Path>, backend: Backend) -> Result<Model, Error>;
    pub fn infer(&mut self, image_chw: &[f32], ori_h: u32, ori_w: u32)
        -> Result<Output<'_>, Error>; // image_chw.len() must == IMAGE_ELEMS
}

pub struct Output<'a> {                // borrows Model-owned buffers, valid until the next infer
    pub infer_ms: f64,
}
impl Output<'_> {
    pub fn detections(&self) -> impl Iterator<Item = Detection>; // 300 rows, score descending
    pub fn raw_dets(&self)  -> &[f32];   // 300*7
    pub fn mask(&self, i: usize) -> &[i32]; // 200*200 mask of row i
    pub fn raw_masks(&self) -> &[i32];   // 300*200*200
}

pub struct Detection {
    pub label: i32, pub score: f32,
    pub x1: f32, pub y1: f32, pub x2: f32, pub y2: f32,
    pub order: i32,
}

// Constants: IMG_SIZE=800, IMAGE_ELEMS=3*800*800, NUM_DETS=300, DET_FIELDS=7, MASK_HW=200, MASK_PIX=40000
```

The borrow checker enforces buffer safety: `infer` cannot be called again while an `Output` is alive.

### Example

```rust
use ppdoclayout::{Backend, Model};

let mut model = Model::new("model-f32.gguf", Backend::Cuda)?;
let image: Vec<f32> = load_and_preprocess("doc.jpg"); // CHW 3x800x800, /255
let out = model.infer(&image, 2339, 1654)?;
println!("{:.1} ms", out.infer_ms);
for (i, det) in out.detections().enumerate() {
    if det.score > 0.5 {
        let mask = out.mask(i);
        println!("{:?} mask_px={}", det, mask.iter().sum::<i32>());
    }
}
```

See `rust/ppdoclayout/examples/detect.rs` for a complete runnable example.

---

## GPU support (including RTX 50 / Blackwell)

- The vendored ggml (v0.17.0) ships with Blackwell support (`GGML_CUDA_CC_BLACKWELL = 1200`). With a CUDA ≥ 12.8 toolchain, **the default architecture list already includes `120a-real` (RTX 50) and `121a-real` when `CMAKE_CUDA_ARCHITECTURES` is not specified** — a fresh configure supports Blackwell natively.
- The existing `build-cuda/` in this repo was configured with `-DCMAKE_CUDA_ARCHITECTURES=89` (RTX 40). The suffix-less `89` also embeds PTX, so it can run on RTX 50 via PTX JIT, but reconfiguring on a 50-series machine is recommended:

```bat
rmdir /s /q build-cuda
cmake -S . -B build-cuda -DUSE_CUDA=ON            rem use ggml's default archs (includes 120a-real)
rem or specify explicitly:
cmake -S . -B build-cuda -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real
```

- The numerical-parity patches are architecture-independent and equally effective on Blackwell (ggml is a git submodule; the patches live in `patches/ggml-fp32-parity.patch` and are **applied automatically and idempotently at cmake configure time** — no manual steps required):
  - `ggml/src/ggml-cuda/common.cuh`: force `CUBLAS_DEFAULT_MATH` for cuBLAS (disables TF32);
  - `ggml/src/ggml-cuda/mmf.cu`: disable the F32 mmf (TF32 MMA) path, effective for all `cc >= Ampere` (including Blackwell).

## Known limitations

- Input is fixed at 800×800, batch=1 (same as the official deployment model).
- Errors are printed to stderr; the API only returns error codes / NULL.
- CPU thread count uses the ggml default (all physical cores); no API to adjust it yet.
- `ppdl_dump` is for debugging only (exports intermediate activations for comparison with `scripts/cmp.py`).
