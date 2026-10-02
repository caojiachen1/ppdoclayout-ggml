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

// micro benchmark: one k3 conv (IC 192 -> OC 128 at 200x200) through the exact
// graph shape the model builder uses: F32 im2col -> mul_mat -> permute -> cont

int main(int argc, char ** argv) {
    const char * model_path = "model-f32.gguf";
    const char * input_path = nullptr;
    const char * out_dir    = ".";
    const char * dump_dir   = nullptr;
    const char * dump_raw_pfx = nullptr;
    const char * batch_path = nullptr;
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
    else if (a == "--dump-raw") dump_raw_pfx = next();
    else if (a == "--batch")     batch_path  = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (!input_path && !batch_path) { fprintf(stderr, "usage: ppdoclayout -m model.gguf -i input.bin --ori-h H --ori-w W -o outdir [--backend cpu|cuda] [--bench N] [--dump-dir d] [--dump-raw pfx] | --batch manifest\n"); return 1; }

    const int backend = backend_name == "cuda" ? PPDL_BACKEND_CUDA : PPDL_BACKEND_CPU;
    ppdl_ctx * ctx = ppdl_init_ex(model_path, backend, dump_dir != nullptr);
    if (!ctx) return 1;

    // batch mode: manifest lines "in.bin|outdir|ori_h|ori_w", model loaded once
    if (batch_path) {
        FILE * mf = fopen(batch_path, "r");
        if (!mf) { fprintf(stderr, "cannot open %s\n", batch_path); return 1; }
        char line[1024];
        int idx = 0, fails = 0;
        while (fgets(line, sizeof line, mf)) {
            char in[900]; int bh = 0, bw = 0; char od[900] = ".";
            if (sscanf(line, "%899[^|]|%899[^|]|%d|%d", in, od, &bh, &bw) != 4) continue;
            std::vector<float> img((size_t) 3 * PPDL_IMG_SIZE * PPDL_IMG_SIZE);
            if (!read_file(in, img.data(), img.size() * 4)) { fprintf(stderr, "read fail %s\n", in); fails++; continue; }
            MKDIR(od);
            ppdl_result res;
            if (ppdl_infer(ctx, img.data(), bh, bw, &res) != 0) { fprintf(stderr, "infer fail %s\n", in); fails++; continue; }
            const std::string odd = od;
            write_file(odd + "/out0.bin", res.dets, (size_t) res.num_dets * PPDL_DET_FIELDS * 4);
            write_file(odd + "/out2.bin", res.masks, (size_t) res.num_dets * PPDL_MASK_HW * PPDL_MASK_HW * 4);
            idx++;
        }
        fclose(mf);
        fprintf(stderr, "batch: %d done, %d failed\n", idx, fails);
        ppdl_free(ctx);
        return fails ? 1 : 0;
    }

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
    if (dump_raw_pfx) ppdl_dump_raw(ctx, dump_raw_pfx);

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
