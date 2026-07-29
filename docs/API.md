# PP-DocLayoutV3 ggml — C/C++ 与 Rust API 使用文档

[English version](API.en.md)

用 ggml 实现的 PP-DocLayoutV3 版面分析推理库。输出与官方 ONNX 导出逐位对齐（label 一致、score 误差 < 1e-6、box 误差 0 px、mask IoU ≈ 1.0）。

## 交付物

| 文件 | 说明 |
|---|---|
| `include/ppdoclayout.h` | C API 头文件（C/C++ 通用） |
| `ppdoclayout_c.dll` / `.lib` | 共享库（`build-*/Release/` 下） |
| `rust/ppdoclayout/` | Rust crate（safe 封装 + example） |
| `model-f32.gguf` | 模型权重（由 `scripts/convert.py` 从 safetensors 生成） |

运行时依赖：`ppdoclayout_c.dll` 及同目录的 ggml DLL（`ggml.dll`、`ggml-base.dll`、`ggml-cpu.dll`，CUDA 版另有 `ggml-cuda.dll`）。exe/进程需能找到它们（同目录或 `PATH`）。

## 构建

```bat
rem CPU 版
cmake -S . -B build-cpu -DUSE_CUDA=OFF
cmake --build build-cpu --config Release -j

rem CUDA 版
cmake -S . -B build-cuda -DUSE_CUDA=ON
cmake --build build-cuda --config Release -j

rem MSVC 把 ggml DLL 放在 bin\Release，拷到 exe 旁边
copy /Y build-cuda\bin\Release\*.dll build-cuda\Release\
```

首次 configure 会自动初始化 ggml submodule 并打上 FP32 一致性补丁（见下文 GPU 支持一节）。

## 输入约定（两种语言相同）

调用方负责预处理：

1. 原图 resize 到 **800×800**（测试用 `cv2.INTER_CUBIC`，插值方式不影响 API 正确性，但要和参照系一致）；
2. 像素值 **/255**，不做 mean/std 归一化；
3. 排列为 **CHW float32**（3×800×800，RGB），共 1,920,000 个 float。

`ori_h` / `ori_w` 传原图尺寸，输出框会自动换算回原图像素坐标。

## 输出约定

每次推理固定返回 **300 行检测**，按 score 降序：

- `dets`：`300 × 7` 的 f32 矩阵，每行 `[label, score, x1, y1, x2, y2, order]`；坐标为原图像素（未 clip，可越界）；`order` 是该元素在全部 300 个 query 中的阅读顺序名次（0 = 最先读）。
- `masks`：`300 × 200 × 200` 的 i32（0/1），第 i 行 mask 对应 dets 第 i 行；mask 覆盖 800×800 网络输入（stride 4），映射回原图时按比例缩放即可。
- 无阈值过滤、无 NMS（与 ONNX 图内行为一致）。典型用法是取 `score > 0.5` 的行。

### 25 类标签

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

（部分 id 语义重复，为官方 checkpoint 自带映射。）

---

## C / C++ API

### 接口一览

```c
#include "ppdoclayout.h"

ppdl_ctx * ppdl_init   (const char * model_path, int backend);          // backend: PPDL_BACKEND_CPU / PPDL_BACKEND_CUDA
ppdl_ctx * ppdl_init_ex(const char * model_path, int backend, int enable_dumps);
int        ppdl_infer  (ppdl_ctx *, const float * image_chw, int ori_h, int ori_w, ppdl_result *);
int        ppdl_dump   (ppdl_ctx *, const char * dir);                  // 需 init_ex(enable_dumps=1)，调试用
void       ppdl_free   (ppdl_ctx *);
```

```c
typedef struct {
    const float   * dets;     // [300*7]
    int32_t         num_dets; // 恒为 300
    const int32_t * masks;    // [300*200*200]
    double          infer_ms; // 本次图计算耗时（不含预/后处理）
} ppdl_result;
```

### 语义与约束

- **返回值**：`ppdl_init*` 失败返回 NULL；`ppdl_infer` / `ppdl_dump` 成功返回 0。错误详情打印到 stderr。
- **生命周期**：`ppdl_result` 中的指针指向 ctx 内部缓冲，**下一次 `ppdl_infer` 或 `ppdl_free` 后失效**；不要 free 它们，需要保留就自行拷贝。
- **线程安全**：单个 `ppdl_ctx` 不可并发调用；多线程请每线程建各自的 ctx（每个 ctx 独立加载权重，约 500 MB 内存/显存）。
- **复用**：ctx 可反复 infer（图已预分配，第二次起无额外分配开销）。
- CUDA 后端下 3 个自定义算子（topk / mask_to_box / 可变形注意力采样）自动回落 CPU，由 sched 处理搬运，无需关心。

### 完整示例

```c
#include "ppdoclayout.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    ppdl_ctx * ctx = ppdl_init("model-f32.gguf", PPDL_BACKEND_CUDA);
    if (!ctx) return 1;

    float * img = malloc(sizeof(float) * 3 * PPDL_IMG_SIZE * PPDL_IMG_SIZE);
    // ... 填入 CHW RGB /255 数据（例如从文件读取或自行 resize）...

    ppdl_result res;
    if (ppdl_infer(ctx, img, /*ori_h=*/2339, /*ori_w=*/1654, &res) != 0) return 1;
    printf("infer %.1f ms\n", res.infer_ms);

    for (int i = 0; i < res.num_dets; i++) {
        const float * d = res.dets + i * PPDL_DET_FIELDS;
        if (d[1] < 0.5f) continue;
        printf("label=%d score=%.3f box=(%.1f,%.1f,%.1f,%.1f) order=%d\n",
               (int) d[0], d[1], d[2], d[3], d[4], d[5], (int) d[6]);
        const int32_t * mask = res.masks + (size_t) i * PPDL_MASK_HW * PPDL_MASK_HW;
        (void) mask; // 200x200 0/1
    }
    ppdl_free(ctx);
    free(img);
    return 0;
}
```

### CMake 集成

```cmake
add_executable(myapp main.c)
target_include_directories(myapp PRIVATE path/to/ppdoclayout-ggml/include)
target_link_libraries(myapp PRIVATE path/to/build-cuda/Release/ppdoclayout_c.lib)
# 运行时把 ppdoclayout_c.dll + ggml*.dll 放到 myapp.exe 旁
```

---

## Rust API

crate 位于 `rust/ppdoclayout`，通过 FFI 链接 `ppdoclayout_c`。

### 引入

```toml
[dependencies]
ppdoclayout = { path = "path/to/ppdoclayout-ggml/rust/ppdoclayout" }
```

链接库定位（`build.rs`）：优先读环境变量 `PPDL_LIB_DIR`，否则自动尝试仓库内 `build-cuda/Release`、`build-cpu/Release`。运行时同样需要 DLL 在 `PATH`：

```bat
set PPDL_LIB_DIR=D:\Codebase\ppdoclayout-ggml\build-cuda\Release
set PATH=%PPDL_LIB_DIR%;%PATH%
cargo run --release --example detect -- model-f32.gguf ref\input.bin cuda 2339 1654
```

### API 一览

```rust
pub enum Backend { Cpu, Cuda }

pub struct Model;                      // Send（可移动到其他线程），非 Sync
impl Model {
    pub fn new(path: impl AsRef<Path>, backend: Backend) -> Result<Model, Error>;
    pub fn infer(&mut self, image_chw: &[f32], ori_h: u32, ori_w: u32)
        -> Result<Output<'_>, Error>; // image_chw.len() 必须 == IMAGE_ELEMS
}

pub struct Output<'a> {                // 借用 Model 内部缓冲，下次 infer 前有效
    pub infer_ms: f64,
}
impl Output<'_> {
    pub fn detections(&self) -> impl Iterator<Item = Detection>; // 300 行，score 降序
    pub fn raw_dets(&self)  -> &[f32];   // 300*7
    pub fn mask(&self, i: usize) -> &[i32]; // 第 i 行的 200*200 mask
    pub fn raw_masks(&self) -> &[i32];   // 300*200*200
}

pub struct Detection {
    pub label: i32, pub score: f32,
    pub x1: f32, pub y1: f32, pub x2: f32, pub y2: f32,
    pub order: i32,
}

// 常量：IMG_SIZE=800, IMAGE_ELEMS=3*800*800, NUM_DETS=300, DET_FIELDS=7, MASK_HW=200, MASK_PIX=40000
```

借用检查保证了缓冲安全：`Output` 存活期间无法再次调用 `infer`。

### 示例

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

完整可运行示例见 `rust/ppdoclayout/examples/detect.rs`。

---

## GPU 支持（含 RTX 50 / Blackwell）

- vendored ggml（v0.17.0）自带 Blackwell 支持（`GGML_CUDA_CC_BLACKWELL = 1200`），配合 CUDA ≥ 12.8 的工具链，**不指定 `CMAKE_CUDA_ARCHITECTURES` 时默认架构列表已包含 `120a-real`（RTX 50）与 `121a-real`**，即全新 configure 直接原生支持 Blackwell。
- 本仓库现有 `build-cuda/` 是用 `-DCMAKE_CUDA_ARCHITECTURES=89`（RTX 40）配置的；无后缀的 `89` 同时嵌入 PTX，RTX 50 上可经 PTX JIT 运行，但推荐在 50 系机器上重新 configure：

```bat
rmdir /s /q build-cuda
cmake -S . -B build-cuda -DUSE_CUDA=ON            rem 用 ggml 默认架构（含 120a-real）
rem 或者显式指定：
cmake -S . -B build-cuda -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real
```

- 数值一致性补丁与架构无关，Blackwell 上同样生效（ggml 为 submodule，补丁存于 `patches/ggml-fp32-parity.patch`，**cmake configure 时自动初始化 submodule 并幂等打补丁**，无需手动操作）：
  - `ggml/src/ggml-cuda/common.cuh`：cuBLAS 强制 `CUBLAS_DEFAULT_MATH`（禁 TF32）；
  - `ggml/src/ggml-cuda/mmf.cu`：禁用 F32 mmf（TF32 MMA）路径，对所有 `cc >= Ampere`（含 Blackwell）生效。

## 已知限制

- 输入固定 800×800、batch=1（与官方部署模型一致）。
- 错误信息输出到 stderr，API 只返回错误码/NULL。
- CPU 线程数使用 ggml 默认值（全部物理核），暂无 API 调节。
- `ppdl_dump` 仅用于调试（导出中间激活与 `scripts/cmp.py` 比对）。
