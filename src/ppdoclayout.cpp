// PP-DocLayoutV3 inference with ggml.
// Weights: model-f32.gguf (BN folded, produced by convert.py).
// Output binary layout matches the official ONNX export's three fetches.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MKDIR(p) mkdir(p, 0755)
#endif

// ---------------------------------------------------------------- constants

static const int IMG        = 800;
static const int NQ         = 300;   // num queries
static const int NC         = 25;    // num classes
static const int DM         = 256;   // d_model
static const int NH         = 8;     // heads
static const int HD         = 32;    // head dim
static const int NLVL       = 3;
static const int NPTS       = 4;
static const int MASK_HW    = 200;
static const int MASK_PIX   = MASK_HW * MASK_HW;
static const int LVL_W[3]   = {100, 50, 25};
static const int LVL_H[3]   = {100, 50, 25};
static const int LVL_START[3] = {0, 10000, 12500};
static const int NTOK       = 13125;

struct dump_entry { std::string name; ggml_tensor * t; };

// ---------------------------------------------------------------- custom ops
// all run on CPU; sched migrates inputs automatically

// src0: class logits ne(25, 13125) -> dst i32 ne(300): indices of top-300
// queries by max-class logit, descending, ties -> lower index first
static void op_topk300(ggml_tensor * dst, int ith, int nth, void *) {
    if (ith != 0) return;
    const ggml_tensor * s = dst->src[0];
    const float * lg = (const float *) s->data;
    std::vector<float> mx(NTOK);
    for (int q = 0; q < NTOK; q++) {
        const float * p = lg + (size_t) q * NC;
        float m = p[0];
        for (int c = 1; c < NC; c++) m = std::max(m, p[c]);
        mx[q] = m;
    }
    std::vector<int> idx(NTOK);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + NQ, idx.end(), [&](int a, int b) {
        if (mx[a] != mx[b]) return mx[a] > mx[b];
        return a < b;
    });
    int32_t * out = (int32_t *) dst->data;
    for (int i = 0; i < NQ; i++) out[i] = idx[i];
}

// src0: mask logits ne(40000, 300) -> dst f32 ne(4, 300):
// inverse_sigmoid(mask_to_box(logit > 0)), cxcywh normalized by 200
static void op_mask_box(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * s = dst->src[0];
    const float * ml = (const float *) s->data;
    float * out = (float *) dst->data;
    auto invsig = [](float x) {
        x = std::min(std::max(x, 0.0f), 1.0f);
        float x1 = std::max(x, 1e-5f);
        float x2 = std::max(1.0f - x, 1e-5f);
        return std::log(x1 / x2);
    };
    for (int q = ith; q < NQ; q += nth) {
        const float * m = ml + (size_t) q * MASK_PIX;
        float xmin = FLT_MAX, ymin = FLT_MAX, xmax = -FLT_MAX, ymax = -FLT_MAX;
        bool any = false;
        for (int y = 0; y < MASK_HW; y++) {
            const float * row = m + (size_t) y * MASK_HW;
            for (int x = 0; x < MASK_HW; x++) {
                if (row[x] > 0.0f) {
                    any = true;
                    xmin = std::min(xmin, (float) x);
                    xmax = std::max(xmax, (float) x);
                    ymin = std::min(ymin, (float) y);
                    ymax = std::max(ymax, (float) y);
                }
            }
        }
        float bx0, by0, bx1, by1;
        if (!any) {
            bx0 = by0 = bx1 = by1 = 0.0f;
        } else {
            bx0 = xmin; by0 = ymin; bx1 = xmax + 1.0f; by1 = ymax + 1.0f;
        }
        bx0 /= MASK_HW; bx1 /= MASK_HW; by0 /= MASK_HW; by1 /= MASK_HW;
        float cx = (bx0 + bx1) * 0.5f, cy = (by0 + by1) * 0.5f;
        float bw = bx1 - bx0, bh = by1 - by0;
        float * o = out + (size_t) q * 4;
        o[0] = invsig(cx); o[1] = invsig(cy); o[2] = invsig(bw); o[3] = invsig(bh);
    }
}

// multiscale deformable attention
// src0 value ne(256,13125)  [t*256 + h*32 + c]
// src1 offsets ne(192,300)  [q*192 + ((h*3+l)*4+p)*2 + xy]
// src2 attn weights (pre-softmax) ne(96,300) [q*96 + h*12 + l*4 + p]
// src3 ref cxcywh sigmoid space ne(4,300)
// dst ne(256,300)
static void op_msdeform(ggml_tensor * dst, int ith, int nth, void *) {
    const float * val = (const float *) dst->src[0]->data;
    const float * off = (const float *) dst->src[1]->data;
    const float * awr = (const float *) dst->src[2]->data;
    const float * ref = (const float *) dst->src[3]->data;
    float * out = (float *) dst->data;

    for (int q = ith; q < NQ; q += nth) {
        const float * rq = ref + (size_t) q * 4;
        float * oq = out + (size_t) q * DM;
        memset(oq, 0, DM * sizeof(float));
        for (int h = 0; h < NH; h++) {
            // softmax over 12 (levels*points)
            const float * a = awr + (size_t) q * 96 + h * 12;
            float amax = a[0];
            for (int i = 1; i < 12; i++) amax = std::max(amax, a[i]);
            float w[12], wsum = 0.0f;
            for (int i = 0; i < 12; i++) { w[i] = std::exp(a[i] - amax); wsum += w[i]; }
            for (int i = 0; i < 12; i++) w[i] /= wsum;

            for (int l = 0; l < NLVL; l++) {
                const int W = LVL_W[l], H = LVL_H[l], st = LVL_START[l];
                for (int p = 0; p < NPTS; p++) {
                    const float ox = off[(size_t) q * 192 + (((h * 3 + l) * 4 + p) * 2 + 0)];
                    const float oy = off[(size_t) q * 192 + (((h * 3 + l) * 4 + p) * 2 + 1)];
                    const float lx = rq[0] + ox / NPTS * rq[2] * 0.5f;
                    const float ly = rq[1] + oy / NPTS * rq[3] * 0.5f;
                    // grid_sample bilinear, zeros pad, align_corners=false
                    const float fx = lx * W - 0.5f;
                    const float fy = ly * H - 0.5f;
                    const int x0 = (int) std::floor(fx), y0 = (int) std::floor(fy);
                    const float tx = fx - x0, ty = fy - y0;
                    const float wt = w[l * 4 + p];
                    const float cw[4] = {
                        (1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty };
                    const int xs[4] = { x0, x0 + 1, x0, x0 + 1 };
                    const int ys[4] = { y0, y0, y0 + 1, y0 + 1 };
                    for (int k = 0; k < 4; k++) {
                        const int xx = xs[k], yy = ys[k];
                        if (xx < 0 || xx >= W || yy < 0 || yy >= H || cw[k] == 0.0f) continue;
                        const float * v = val + ((size_t) (st + yy * W + xx) * DM) + h * HD;
                        const float ww = wt * cw[k];
                        for (int c = 0; c < HD; c++) oq[h * HD + c] += ww * v[c];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- model

struct model {
    ggml_context * ctx_w = nullptr;
    gguf_context * gctx  = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;

    ggml_tensor * get(const std::string & name) const {
        ggml_tensor * t = ggml_get_tensor(ctx_w, name.c_str());
        if (!t) { fprintf(stderr, "missing tensor: %s\n", name.c_str()); exit(1); }
        return t;
    }
};

static bool load_model(model & m, const char * path, ggml_backend_t backend) {
    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx = &m.ctx_w;
    m.gctx = gguf_init_from_file(path, gp);
    if (!m.gctx) return false;
    m.wbuf = ggml_backend_alloc_ctx_tensors(m.ctx_w, backend);
    if (!m.wbuf) return false;

    FILE * f = fopen(path, "rb");
    if (!f) return false;
    const size_t data_off = gguf_get_data_offset(m.gctx);
    const int64_t n = gguf_get_n_tensors(m.gctx);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(m.gctx, i);
        ggml_tensor * t = ggml_get_tensor(m.ctx_w, name);
        const size_t nb = ggml_nbytes(t);
        buf.resize(nb);
#if defined(_WIN32)
        _fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(m.gctx, i)), SEEK_SET);
#else
        fseeko(f, (off_t) (data_off + gguf_get_tensor_offset(m.gctx, i)), SEEK_SET);
#endif
        if (fread(buf.data(), 1, nb, f) != nb) { fclose(f); return false; }
        ggml_backend_tensor_set(t, buf.data(), 0, nb);
    }
    fclose(f);
    return true;
}

// ---------------------------------------------------------------- graph build

struct graph_out {
    ggml_tensor * image  = nullptr; // input ne(800,800,3,1)
    ggml_tensor * pos    = nullptr; // input ne(256,625)
    ggml_tensor * valid  = nullptr; // input ne(1,13125)
    ggml_tensor * logits = nullptr; // ne(25,300)
    ggml_tensor * boxes  = nullptr; // ne(4,300) sigmoid cxcywh
    ggml_tensor * order  = nullptr; // ne(300,300) raw q@k^T/8, [j + 300*i]
    ggml_tensor * masks  = nullptr; // ne(40000,300) logits
    std::vector<dump_entry> dumps;
};

enum act_t { ACT_NONE, ACT_RELU, ACT_SILU };

struct builder {
    ggml_context * ctx;
    const model  * m;
    graph_out    * out;
    bool           want_dumps;

    void dump(const char * name, ggml_tensor * t) {
        if (!want_dumps) return;
        ggml_set_output(t);
        out->dumps.push_back({name, t});
    }

    ggml_tensor * act(ggml_tensor * x, act_t a) {
        switch (a) {
            case ACT_RELU: return ggml_relu(ctx, x);
            case ACT_SILU: return ggml_silu(ctx, x);
            default:       return x;
        }
    }

    // conv with F32 im2col (ggml_conv_2d uses F16 im2col -> precision loss)
    ggml_tensor * conv(ggml_tensor * x, const std::string & name, int s, int p, act_t a) {
        ggml_tensor * w = m->get(name + ".weight");
        ggml_tensor * b = m->get(name + ".bias");
        ggml_tensor * r;
        if (w->ne[2] == 1 && x->ne[2] != 1) { // depthwise
            r = ggml_conv_2d_dw_direct(ctx, w, x, s, s, p, p, 1, 1);
        } else {
            ggml_tensor * im = ggml_im2col(ctx, w, x, s, s, p, p, 1, 1, true, GGML_TYPE_F32);
            r = ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2] * im->ne[3]),
                ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
            r = ggml_reshape_4d(ctx, r, im->ne[1], im->ne[2], im->ne[3], w->ne[3]);
            r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
        }
        r = ggml_add(ctx, r, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
        return act(r, a);
    }

    ggml_tensor * lin(ggml_tensor * x, const std::string & name) {
        ggml_tensor * r = ggml_mul_mat(ctx, m->get(name + ".weight"), x);
        return ggml_add(ctx, r, m->get(name + ".bias"));
    }

    ggml_tensor * ln(ggml_tensor * x, const std::string & name) {
        ggml_tensor * r = ggml_norm(ctx, x, 1e-5f);
        r = ggml_mul(ctx, r, m->get(name + ".weight"));
        return ggml_add(ctx, r, m->get(name + ".bias"));
    }

    // 3-layer MLP head: lin relu lin relu lin
    ggml_tensor * mlp3(ggml_tensor * x, const std::string & pfx) {
        x = ggml_relu(ctx, lin(x, pfx + ".0"));
        x = ggml_relu(ctx, lin(x, pfx + ".1"));
        return lin(x, pfx + ".2");
    }

    // image ne(W,H,C,1) -> sequence ne(C, W*H), token = y*W + x
    ggml_tensor * to_seq(ggml_tensor * x) {
        ggml_tensor * t = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
        return ggml_reshape_2d(ctx, t, t->ne[0], t->ne[1] * t->ne[2]);
    }

    // MHA, 8 heads x 32: q,k from qk_in, v from v_in. all ne(256,T)
    ggml_tensor * mha(ggml_tensor * qk_in, ggml_tensor * v_in, const std::string & pfx) {
        const int64_t T = qk_in->ne[1];
        ggml_tensor * q = lin(qk_in, pfx + ".q");
        ggml_tensor * k = lin(qk_in, pfx + ".k");
        ggml_tensor * v = lin(v_in,  pfx + ".v");
        q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, HD, NH, T), 0, 2, 1, 3); // (32,T,8)
        k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, HD, NH, T), 0, 2, 1, 3);
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                            // (Tk,Tq,8)
        kq = ggml_soft_max(ctx, ggml_scale(ctx, kq, 1.0f / std::sqrt((float) HD)));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, v, HD, NH, T), 1, 2, 0, 3));                  // (T,32,8)
        ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);                         // (32,Tq,8)
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));              // (32,8,Tq)
        kqv = ggml_reshape_2d(ctx, kqv, DM, T);
        return lin(kqv, pfx + ".o");
    }

    ggml_tensor * inv_sigmoid(ggml_tensor * x) {
        x = ggml_clamp(ctx, x, 0.0f, 1.0f);
        ggml_tensor * x1 = ggml_clamp(ctx, x, 1e-5f, 1.0f);
        ggml_tensor * x2 = ggml_clamp(ctx, ggml_scale_bias(ctx, x, -1.0f, 1.0f), 1e-5f, 1.0f);
        return ggml_sub(ctx, ggml_log(ctx, x1), ggml_log(ctx, x2));
    }

    // ---- backbone

    ggml_tensor * hg_block(ggml_tensor * x, const std::string & pfx, bool light) {
        ggml_tensor * cat = x;
        ggml_tensor * cur = x;
        for (int l = 0; l < 6; l++) {
            const std::string lp = pfx + ".l" + std::to_string(l);
            if (light) {
                cur = conv(cur, lp + ".c1", 1, 0, ACT_NONE);
                cur = conv(cur, lp + ".c2", 1, 2, ACT_RELU); // dw k5 p2
            } else {
                cur = conv(cur, lp, 1, 1, ACT_RELU);          // k3 p1
            }
            cat = ggml_concat(ctx, cat, cur, 2);
        }
        ggml_tensor * y = conv(cat, pfx + ".agg0", 1, 0, ACT_RELU);
        y = conv(y, pfx + ".agg1", 1, 0, ACT_RELU);
        return y;
    }

    void backbone(ggml_tensor * img, ggml_tensor ** x4, ggml_tensor ** p8,
                  ggml_tensor ** p16, ggml_tensor ** p32) {
        ggml_tensor * x = conv(img, "stem1", 2, 1, ACT_RELU);          // 400,32
        dump("stem1", x);
        ggml_tensor * xp = ggml_pad(ctx, x, 1, 1, 0, 0);               // 401
        ggml_tensor * a = conv(xp, "stem2a", 1, 0, ACT_RELU);          // 400,16
        a = ggml_pad(ctx, a, 1, 1, 0, 0);
        a = conv(a, "stem2b", 1, 0, ACT_RELU);                          // 400,32
        ggml_tensor * pl = ggml_pool_2d(ctx, xp, GGML_OP_POOL_MAX, 2, 2, 1, 1, 0, 0); // 400,32
        x = ggml_concat(ctx, pl, a, 2);                                 // 400,64
        x = conv(x, "stem3", 2, 1, ACT_RELU);                           // 200,32
        x = conv(x, "stem4", 1, 0, ACT_RELU);                           // 200,48
        dump("stem", x);

        // stage0: no downsample, 1 block, no residual
        ggml_tensor * s0 = hg_block(x, "st0.b0", false);                // 200,128
        dump("st0", s0);
        // stage1
        ggml_tensor * s1 = conv(s0, "st1.ds", 2, 1, ACT_NONE);          // 100 dw
        s1 = hg_block(s1, "st1.b0", false);                             // 100,512
        dump("st1", s1);
        // stage2: 3 blocks, light, residual on b1/b2
        ggml_tensor * s2 = conv(s1, "st2.ds", 2, 1, ACT_NONE);          // 50
        s2 = hg_block(s2, "st2.b0", true);                              // 50,1024
        s2 = ggml_add(ctx, hg_block(s2, "st2.b1", true), s2);
        s2 = ggml_add(ctx, hg_block(s2, "st2.b2", true), s2);
        dump("st2", s2);
        // stage3
        ggml_tensor * s3 = conv(s2, "st3.ds", 2, 1, ACT_NONE);          // 25
        s3 = hg_block(s3, "st3.b0", true);                              // 25,2048
        dump("st3", s3);

        *x4 = s0; *p8 = s1; *p16 = s2; *p32 = s3;
    }

    // ---- hybrid encoder

    ggml_tensor * repvgg(ggml_tensor * x, const std::string & pfx) {
        ggml_tensor * y = ggml_add(ctx,
            conv(x, pfx + ".c1", 1, 1, ACT_NONE),
            conv(x, pfx + ".c2", 1, 0, ACT_NONE));
        return ggml_silu(ctx, y);
    }

    ggml_tensor * csp(ggml_tensor * x, const std::string & pfx) {
        ggml_tensor * a = conv(x, pfx + ".c1", 1, 0, ACT_SILU);
        for (int i = 0; i < 3; i++) a = repvgg(a, pfx + ".bn" + std::to_string(i));
        ggml_tensor * b = conv(x, pfx + ".c2", 1, 0, ACT_SILU);
        return ggml_add(ctx, a, b);
    }

    ggml_tensor * interp(ggml_tensor * x, int w, int h, uint32_t mode) {
        return ggml_interpolate(ctx, x, w, h, x->ne[2], x->ne[3], mode);
    }

    void encoder(ggml_tensor * x4, ggml_tensor * p8r, ggml_tensor * p16r, ggml_tensor * p32r,
                 ggml_tensor * pos, ggml_tensor ** f8o, ggml_tensor ** n16o,
                 ggml_tensor ** n32o, ggml_tensor ** mfeat) {
        ggml_tensor * p8  = conv(p8r,  "enc_in0", 1, 0, ACT_NONE); // (100,100,256)
        ggml_tensor * p16 = conv(p16r, "enc_in1", 1, 0, ACT_NONE); // (50,50,256)
        ggml_tensor * p32 = conv(p32r, "enc_in2", 1, 0, ACT_NONE); // (25,25,256)

        // AIFI on p32 (post-norm)
        {
            ggml_tensor * s = to_seq(p32);                       // (256,625)
            ggml_tensor * qk = ggml_add(ctx, s, pos);
            ggml_tensor * at = mha(qk, s, "aifi");
            s = ln(ggml_add(ctx, s, at), "aifi.ln1");
            ggml_tensor * ff = lin(ggml_gelu_erf(ctx, lin(s, "aifi.fc1")), "aifi.fc2");
            s = ln(ggml_add(ctx, s, ff), "aifi.ln2");
            // back to image (25,25,256)
            ggml_tensor * t = ggml_reshape_3d(ctx, s, DM, 25, 25);
            p32 = ggml_cont(ctx, ggml_permute(ctx, t, 2, 0, 1, 3));
            dump("aifi", p32);
        }

        // FPN (top-down); laterals replace the stored top maps
        ggml_tensor * lat0 = conv(p32, "lat0", 1, 0, ACT_SILU);                 // 25
        ggml_tensor * up0  = interp(lat0, 50, 50, GGML_SCALE_MODE_NEAREST);
        ggml_tensor * f16  = csp(ggml_concat(ctx, up0, p16, 2), "fpn0");        // 50
        ggml_tensor * lat1 = conv(f16, "lat1", 1, 0, ACT_SILU);
        ggml_tensor * up1  = interp(lat1, 100, 100, GGML_SCALE_MODE_NEAREST);
        ggml_tensor * f8   = csp(ggml_concat(ctx, up1, p8, 2), "fpn1");         // 100
        dump("f8", f8);

        // PAN (bottom-up) against laterals
        ggml_tensor * d16 = conv(f8, "pand0", 2, 1, ACT_SILU);                  // 50
        ggml_tensor * n16 = csp(ggml_concat(ctx, d16, lat1, 2), "pan0");
        dump("n16", n16);
        ggml_tensor * d32 = conv(n16, "pand1", 2, 1, ACT_SILU);                 // 25
        ggml_tensor * n32 = csp(ggml_concat(ctx, d32, lat0, 2), "pan1");
        dump("n32", n32);

        // mask feature head on [f8, n16, n32]
        ggml_tensor * h0 = conv(f8, "mh0.0", 1, 1, ACT_SILU);                    // 100,64
        ggml_tensor * h1 = conv(n16, "mh1.0", 1, 1, ACT_SILU);
        h1 = interp(h1, 100, 100, GGML_SCALE_MODE_BILINEAR);
        ggml_tensor * h2 = conv(n32, "mh2.0", 1, 1, ACT_SILU);
        h2 = interp(h2, 50, 50, GGML_SCALE_MODE_BILINEAR);
        h2 = conv(h2, "mh2.2", 1, 1, ACT_SILU);
        h2 = interp(h2, 100, 100, GGML_SCALE_MODE_BILINEAR);
        ggml_tensor * mf = ggml_add(ctx, ggml_add(ctx, h0, h1), h2);
        mf = conv(mf, "mh.out", 1, 1, ACT_SILU);                                 // 100,64
        mf = interp(mf, 200, 200, GGML_SCALE_MODE_BILINEAR);                     // 200,64
        mf = ggml_add(ctx, mf, conv(x4, "mlat", 1, 1, ACT_SILU));
        mf = conv(mf, "mout.base", 1, 1, ACT_SILU);
        mf = conv(mf, "mout.conv", 1, 0, ACT_NONE);                              // 200,32
        dump("mask_feat", mf);

        *f8o = f8; *n16o = n16; *n32o = n32; *mfeat = mf;
    }

    // ---- full graph

    void build(graph_out & go, ggml_cgraph * gf) {
        out = &go;

        go.image = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IMG, IMG, 3, 1);
        ggml_set_name(go.image, "image");
        ggml_set_input(go.image);
        go.pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DM, 625);
        ggml_set_name(go.pos, "pos");
        ggml_set_input(go.pos);
        go.valid = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, NTOK);
        ggml_set_name(go.valid, "valid");
        ggml_set_input(go.valid);

        ggml_tensor *x4, *p8, *p16, *p32;
        backbone(go.image, &x4, &p8, &p16, &p32);

        ggml_tensor *f8, *n16, *n32, *mfeat;
        encoder(x4, p8, p16, p32, go.pos, &f8, &n16, &n32, &mfeat);

        // mask_feat flattened: (32, 40000)
        ggml_tensor * mflat = ggml_reshape_2d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, mfeat, 1, 2, 0, 3)), 32, MASK_PIX);

        // decoder input projections + flatten + concat -> source (256,13125)
        ggml_tensor * s0 = to_seq(conv(f8,  "dec_in0", 1, 0, ACT_NONE));
        ggml_tensor * s1 = to_seq(conv(n16, "dec_in1", 1, 0, ACT_NONE));
        ggml_tensor * s2 = to_seq(conv(n32, "dec_in2", 1, 0, ACT_NONE));
        ggml_tensor * source = ggml_concat(ctx, ggml_concat(ctx, s0, s1, 1), s2, 1);
        dump("source", source);

        // two-stage selection
        ggml_tensor * memory = ggml_mul(ctx, source, go.valid);
        ggml_tensor * om = ln(lin(memory, "enc_out.l"), "enc_out.ln");
        ggml_tensor * ecls = lin(om, "score_head");                    // (25,13125)
        ggml_tensor * targs[1] = { ecls };
        ggml_tensor * topk = ggml_custom_4d(ctx, GGML_TYPE_I32, NQ, 1, 1, 1,
                                            targs, 1, op_topk300, 1, nullptr);
        dump("topk_idx", topk);
        ggml_tensor * target = ggml_get_rows(ctx, om, topk);           // (256,300)
        dump("target", target);

        // enhanced reference from encoder masks
        ggml_tensor * outq0 = ln(target, "dec_norm");
        ggml_tensor * mq0 = mlp3(outq0, "mq");                          // (32,300)
        ggml_tensor * encm = ggml_mul_mat(ctx, mflat, ggml_cont(ctx, mq0)); // (40000,300)
        dump("enc_masks", encm);
        ggml_tensor * margs[1] = { encm };
        ggml_tensor * refu = ggml_custom_4d(ctx, GGML_TYPE_F32, 4, NQ, 1, 1,
                                            margs, 1, op_mask_box, GGML_N_TASKS_MAX, nullptr);
        dump("init_ref_unact", refu);
        ggml_tensor * ref = ggml_sigmoid(ctx, refu);

        // decoder layers
        ggml_tensor * x = target;
        for (int i = 0; i < 6; i++) {
            const std::string dp = "dec" + std::to_string(i);
            ggml_tensor * qpos = lin(ggml_relu(ctx, lin(ref, "qpos.0")), "qpos.1");
            // self attention
            ggml_tensor * qk = ggml_add(ctx, x, qpos);
            ggml_tensor * sa = mha(qk, x, dp + ".sa");
            x = ln(ggml_add(ctx, x, sa), dp + ".ln1");
            // cross attention (deformable)
            ggml_tensor * z = ggml_add(ctx, x, qpos);
            ggml_tensor * value = lin(source, dp + ".ca.vp");
            ggml_tensor * offs = lin(z, dp + ".ca.so");
            ggml_tensor * aw = lin(z, dp + ".ca.aw");
            ggml_tensor * cargs[4] = { ggml_cont(ctx, value), ggml_cont(ctx, offs),
                                       ggml_cont(ctx, aw), ggml_cont(ctx, ref) };
            ggml_tensor * ca = ggml_custom_4d(ctx, GGML_TYPE_F32, DM, NQ, 1, 1,
                                              cargs, 4, op_msdeform, GGML_N_TASKS_MAX, nullptr);
            ca = lin(ca, dp + ".ca.op");
            x = ln(ggml_add(ctx, x, ca), dp + ".ln2");
            // ffn
            ggml_tensor * ff = lin(ggml_relu(ctx, lin(x, dp + ".fc1")), dp + ".fc2");
            x = ln(ggml_add(ctx, x, ff), dp + ".ln3");
            dump(("hidden" + std::to_string(i)).c_str(), x);
            // box refinement (tied bbox head)
            ggml_tensor * delta = mlp3(x, "bbox_head");
            ref = ggml_sigmoid(ctx, ggml_add(ctx, delta, inv_sigmoid(ref)));
        }

        // heads on final layer
        ggml_tensor * outq = ln(x, "dec_norm");
        go.logits = lin(outq, "score_head");                            // (25,300)
        go.boxes  = ref;                                                 // (4,300)
        ggml_tensor * mq6 = mlp3(outq, "mq");
        go.masks = ggml_mul_mat(ctx, mflat, ggml_cont(ctx, mq6));        // (40000,300)
        ggml_tensor * oh = lin(outq, "order5");
        ggml_tensor * gp = lin(oh, "gp");                                // (128,300)
        ggml_tensor * qv = ggml_cont(ctx, ggml_view_2d(ctx, gp, 64, NQ, gp->nb[1], 0));
        ggml_tensor * kv = ggml_cont(ctx, ggml_view_2d(ctx, gp, 64, NQ, gp->nb[1], 64 * sizeof(float)));
        go.order = ggml_scale(ctx, ggml_mul_mat(ctx, kv, qv), 1.0f / 8.0f); // (300k,300q)

        ggml_set_output(go.logits);
        ggml_set_output(go.boxes);
        ggml_set_output(go.order);
        ggml_set_output(go.masks);
        ggml_set_name(go.logits, "logits");
        ggml_set_name(go.boxes, "boxes");
        ggml_set_name(go.order, "order_raw");
        ggml_set_name(go.masks, "mask_logits");

        ggml_build_forward_expand(gf, go.logits);
        ggml_build_forward_expand(gf, go.boxes);
        ggml_build_forward_expand(gf, go.order);
        ggml_build_forward_expand(gf, go.masks);
        for (auto & d : go.dumps) ggml_build_forward_expand(gf, d.t);
    }
};

// ---------------------------------------------------------------- host-side constants

static void fill_pos_embed(std::vector<float> & pos) {
    // [sin_h | cos_h | sin_w | cos_w], H-outer, float64 math
    pos.resize((size_t) DM * 625);
    double omega[64];
    for (int i = 0; i < 64; i++) omega[i] = 1.0 / std::pow(10000.0, (double) i / 64.0);
    for (int h = 0; h < 25; h++) {
        for (int w = 0; w < 25; w++) {
            float * p = pos.data() + (size_t) (h * 25 + w) * DM;
            for (int i = 0; i < 64; i++) {
                p[i]       = (float) std::sin(h * omega[i]);
                p[64 + i]  = (float) std::cos(h * omega[i]);
                p[128 + i] = (float) std::sin(w * omega[i]);
                p[192 + i] = (float) std::cos(w * omega[i]);
            }
        }
    }
}

static void fill_valid_mask(std::vector<float> & vm) {
    vm.resize(NTOK);
    const float eps = 1e-2f;
    int t = 0;
    for (int l = 0; l < NLVL; l++) {
        const int W = LVL_W[l], H = LVL_H[l];
        const float wh = 0.05f * (float) std::pow(2.0f, (float) l);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                const float cx = ((float) x + 0.5f) / (float) W;
                const float cy = ((float) y + 0.5f) / (float) H;
                const bool ok = cx > eps && cx < 1.0f - eps &&
                                cy > eps && cy < 1.0f - eps &&
                                wh > eps && wh < 1.0f - eps;
                vm[t++] = ok ? 1.0f : 0.0f;
            }
        }
    }
}

// ---------------------------------------------------------------- postprocess

static void postprocess(const std::vector<float> & logits,  // (300,25) row-major [q*25+c]
                        const std::vector<float> & boxes,   // (300,4)  [q*4+k] cxcywh
                        const std::vector<float> & order,   // [j + 300*i] = q_i . k_j / 8
                        const std::vector<float> & maskl,   // (300,40000)
                        int ori_h, int ori_w,
                        std::vector<float> & out0, std::vector<int32_t> & out2) {
    // reading order votes (Paddle semantics: antisymmetric logits, column sums)
    std::vector<float> votes(NQ, 0.0f);
    for (int j = 0; j < NQ; j++) {
        float acc = 0.0f;
        for (int i = 0; i < NQ; i++) {
            if (i == j) continue;
            const float diff = order[(size_t) j + (size_t) NQ * i] - order[(size_t) i + (size_t) NQ * j];
            acc += 1.0f / (1.0f + std::exp(-diff));
        }
        votes[j] = acc;
    }
    std::vector<int> ptr(NQ);
    std::iota(ptr.begin(), ptr.end(), 0);
    std::stable_sort(ptr.begin(), ptr.end(), [&](int a, int b) { return votes[a] < votes[b]; });
    std::vector<int> order_seq(NQ);
    for (int r = 0; r < NQ; r++) order_seq[ptr[r]] = r;

    // flatten top-300 over 7500 scores
    std::vector<float> scores((size_t) NQ * NC);
    for (int q = 0; q < NQ; q++)
        for (int c = 0; c < NC; c++)
            scores[(size_t) q * NC + c] = 1.0f / (1.0f + std::exp(-logits[(size_t) q * NC + c]));
    std::vector<int> fi((size_t) NQ * NC);
    std::iota(fi.begin(), fi.end(), 0);
    std::partial_sort(fi.begin(), fi.begin() + NQ, fi.end(), [&](int a, int b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    });

    // coordinate restore, ONNX style
    const float sf_h = (float) IMG / (float) ori_h;
    const float sf_w = (float) IMG / (float) ori_w;
    const float oh = std::floor((float) IMG / sf_h + 0.5f);
    const float ow = std::floor((float) IMG / sf_w + 0.5f);

    out0.resize((size_t) NQ * 7);
    out2.assign((size_t) NQ * MASK_PIX, 0);
    for (int r = 0; r < NQ; r++) {
        const int flat = fi[r];
        const int q = flat / NC, c = flat % NC;
        const float * b = boxes.data() + (size_t) q * 4;
        const float x1 = (b[0] - b[2] * 0.5f) * ow;
        const float y1 = (b[1] - b[3] * 0.5f) * oh;
        const float x2 = (b[0] + b[2] * 0.5f) * ow;
        const float y2 = (b[1] + b[3] * 0.5f) * oh;
        float * o = out0.data() + (size_t) r * 7;
        o[0] = (float) c;
        o[1] = scores[flat];
        o[2] = x1; o[3] = y1; o[4] = x2; o[5] = y2;
        o[6] = (float) order_seq[q];
        const float * ml = maskl.data() + (size_t) q * MASK_PIX;
        int32_t * mo = out2.data() + (size_t) r * MASK_PIX;
        for (int p = 0; p < MASK_PIX; p++) mo[p] = ml[p] > 0.0f ? 1 : 0;
    }
}

// ---------------------------------------------------------------- C API

#include "ppdoclayout.h"

struct ppdl_ctx {
    model m;
    std::vector<ggml_backend_t> backends;
    ggml_context * ctx0        = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * gf           = nullptr;
    graph_out go;
    bool computed = false;
    std::vector<float>   out0;
    std::vector<int32_t> out2;
};

static bool write_file(const std::string & path, const void * src, size_t nbytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const size_t wr = fwrite(src, 1, nbytes, f);
    fclose(f);
    return wr == nbytes;
}

extern "C" PPDL_API ppdl_ctx * ppdl_init_ex(const char * model_path, int backend, int enable_dumps) {
    // TF32 matmul on Ampere+ loses ~3 decimal digits; force full FP32 for ONNX parity
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
    ppdl_ctx * c = new ppdl_ctx();

    if (backend == PPDL_BACKEND_CUDA) {
        ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!gpu) { fprintf(stderr, "ppdl: no GPU backend available\n"); delete c; return nullptr; }
        c->backends.push_back(gpu);
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    c->backends.push_back(cpu);

    if (!load_model(c->m, model_path, c->backends[0])) {
        fprintf(stderr, "ppdl: failed to load %s\n", model_path);
        ppdl_free(c);
        return nullptr;
    }
    fprintf(stderr, "ppdl: loaded %s (%lld tensors) on %s\n", model_path,
            (long long) gguf_get_n_tensors(c->m.gctx), ggml_backend_name(c->backends[0]));

    const size_t GRAPH_NODES = 8192;
    const size_t meta_size = 64 * 1024 * (size_t) ggml_tensor_overhead()
                           + ggml_graph_overhead_custom(GRAPH_NODES, false);
    ggml_init_params ip = { meta_size, nullptr, true };
    c->ctx0 = ggml_init(ip);
    c->gf = ggml_new_graph_custom(c->ctx0, GRAPH_NODES, false);

    builder B;
    B.ctx = c->ctx0; B.m = &c->m; B.want_dumps = enable_dumps != 0;
    B.build(c->go, c->gf);
    fprintf(stderr, "ppdl: graph %d nodes\n", ggml_graph_n_nodes(c->gf));

    c->sched = ggml_backend_sched_new(
        c->backends.data(), nullptr, (int) c->backends.size(), GRAPH_NODES, false, true);
    if (!ggml_backend_sched_alloc_graph(c->sched, c->gf)) {
        fprintf(stderr, "ppdl: sched alloc failed\n");
        ppdl_free(c);
        return nullptr;
    }

    std::vector<float> pos, vm;
    fill_pos_embed(pos);
    fill_valid_mask(vm);
    ggml_backend_tensor_set(c->go.pos, pos.data(), 0, pos.size() * 4);
    ggml_backend_tensor_set(c->go.valid, vm.data(), 0, vm.size() * 4);
    return c;
}

extern "C" PPDL_API ppdl_ctx * ppdl_init(const char * model_path, int backend) {
    return ppdl_init_ex(model_path, backend, 0);
}

extern "C" PPDL_API int ppdl_infer(ppdl_ctx * c, const float * image_chw,
                                   int ori_h, int ori_w, ppdl_result * result) {
    if (!c || !image_chw || !result) return 1;
    if (ori_h <= 0 || ori_w <= 0) return 1;

    ggml_backend_tensor_set(c->go.image, image_chw, 0, (size_t) 3 * IMG * IMG * 4);

    const auto t0 = std::chrono::steady_clock::now();
    if (ggml_backend_sched_graph_compute(c->sched, c->gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ppdl: graph compute failed\n");
        return 2;
    }
    const auto t1 = std::chrono::steady_clock::now();
    c->computed = true;

    std::vector<float> logits((size_t) NQ * NC), boxes((size_t) NQ * 4);
    std::vector<float> order((size_t) NQ * NQ), maskl((size_t) NQ * MASK_PIX);
    ggml_backend_tensor_get(c->go.logits, logits.data(), 0, logits.size() * 4);
    ggml_backend_tensor_get(c->go.boxes,  boxes.data(),  0, boxes.size() * 4);
    ggml_backend_tensor_get(c->go.order,  order.data(),  0, order.size() * 4);
    ggml_backend_tensor_get(c->go.masks,  maskl.data(),  0, maskl.size() * 4);

    postprocess(logits, boxes, order, maskl, ori_h, ori_w, c->out0, c->out2);

    result->dets     = c->out0.data();
    result->num_dets = NQ;
    result->masks    = c->out2.data();
    result->infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return 0;
}

extern "C" PPDL_API int ppdl_dump(ppdl_ctx * c, const char * dir) {
    if (!c || !dir) return 1;
    if (c->go.dumps.empty()) { fprintf(stderr, "ppdl: dumps not enabled\n"); return 1; }
    if (!c->computed) { fprintf(stderr, "ppdl: no inference to dump\n"); return 1; }
    MKDIR(dir);
    std::string meta = std::string(dir) + "/shapes.txt";
    FILE * mf = fopen(meta.c_str(), "w");
    if (!mf) return 1;
    for (auto & d : c->go.dumps) {
        std::vector<uint8_t> buf(ggml_nbytes(d.t));
        ggml_backend_tensor_get(d.t, buf.data(), 0, buf.size());
        write_file(std::string(dir) + "/" + d.name + ".bin", buf.data(), buf.size());
        fprintf(mf, "%s %s %lld %lld %lld %lld\n", d.name.c_str(),
                ggml_type_name(d.t->type),
                (long long) d.t->ne[0], (long long) d.t->ne[1],
                (long long) d.t->ne[2], (long long) d.t->ne[3]);
    }
    fclose(mf);
    fprintf(stderr, "ppdl: dumped %zu tensors to %s\n", c->go.dumps.size(), dir);
    return 0;
}

extern "C" PPDL_API void ppdl_free(ppdl_ctx * c) {
    if (!c) return;
    if (c->sched) ggml_backend_sched_free(c->sched);
    if (c->ctx0)  ggml_free(c->ctx0);
    if (c->m.wbuf)  ggml_backend_buffer_free(c->m.wbuf);
    if (c->m.gctx)  gguf_free(c->m.gctx);
    if (c->m.ctx_w) ggml_free(c->m.ctx_w);
    for (auto b : c->backends) ggml_backend_free(b);
    delete c;
}
