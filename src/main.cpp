// CLI for PP-DocLayoutV3 ggml inference, built on the C API.
#include "ppdoclayout.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MKDIR(p) mkdir(p, 0755)
#endif

static bool read_file(const char * path, void * dst, size_t nbytes) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    const size_t rd = fread(dst, 1, nbytes, f);
    fclose(f);
    return rd == nbytes;
}

static bool write_file(const std::string & path, const void * src, size_t nbytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const size_t wr = fwrite(src, 1, nbytes, f);
    fclose(f);
    return wr == nbytes;
}

int main(int argc, char ** argv) {
    const char * model_path = "model-f32.gguf";
    const char * input_path = nullptr;
    const char * out_dir    = ".";
    const char * dump_dir   = nullptr;
    std::string  backend_name = "cpu";
    int ori_h = PPDL_IMG_SIZE, ori_w = PPDL_IMG_SIZE;
    int bench = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")          model_path  = next();
        else if (a == "-i")          input_path  = next();
        else if (a == "-o")          out_dir     = next();
        else if (a == "--ori-h")     ori_h       = atoi(next());
        else if (a == "--ori-w")     ori_w       = atoi(next());
        else if (a == "--backend")   backend_name = next();
        else if (a == "--bench")     bench       = atoi(next());
        else if (a == "--dump-dir")  dump_dir    = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (!input_path) { fprintf(stderr, "usage: ppdoclayout -m model.gguf -i input.bin --ori-h H --ori-w W -o outdir [--backend cpu|cuda] [--bench N] [--dump-dir d]\n"); return 1; }

    const int backend = backend_name == "cuda" ? PPDL_BACKEND_CUDA : PPDL_BACKEND_CPU;
    ppdl_ctx * ctx = ppdl_init_ex(model_path, backend, dump_dir != nullptr);
    if (!ctx) return 1;

    std::vector<float> img((size_t) 3 * PPDL_IMG_SIZE * PPDL_IMG_SIZE);
    if (!read_file(input_path, img.data(), img.size() * 4)) {
        fprintf(stderr, "failed to read %s\n", input_path);
        return 1;
    }

    ppdl_result res;
    if (ppdl_infer(ctx, img.data(), ori_h, ori_w, &res) != 0) return 1;
    fprintf(stderr, "inference: %.1f ms\n", res.infer_ms);

    if (bench > 0) {
        std::vector<double> ts;
        for (int i = 0; i < bench; i++) {
            if (ppdl_infer(ctx, img.data(), ori_h, ori_w, &res) != 0) return 1;
            ts.push_back(res.infer_ms);
        }
        double mean = 0; for (double t : ts) mean += t; mean /= ts.size();
        double var = 0; for (double t : ts) var += (t - mean) * (t - mean);
        var /= ts.size();
        printf("bench_ms mean=%.2f std=%.2f n=%d\n", mean, std::sqrt(var), bench);
    }

    if (dump_dir) ppdl_dump(ctx, dump_dir);

    const int32_t out1 = res.num_dets;
    MKDIR(out_dir);
    const std::string od = out_dir;
    write_file(od + "/out0.bin", res.dets, (size_t) res.num_dets * PPDL_DET_FIELDS * 4);
    write_file(od + "/out1.bin", &out1, 4);
    write_file(od + "/out2.bin", res.masks,
               (size_t) res.num_dets * PPDL_MASK_HW * PPDL_MASK_HW * 4);
    fprintf(stderr, "wrote %s/out0.bin out1.bin out2.bin\n", out_dir);

    ppdl_free(ctx);
    return 0;
}
