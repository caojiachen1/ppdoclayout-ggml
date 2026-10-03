// Standalone check of the Winograd F(2,3) pipeline ops vs a direct conv.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

#define CK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
#define CB(x) do { auto s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS err %d @%d\n", (int) s_, __LINE__); return 1; } } while (0)

__constant__ float WGB[4][4] = {
    { 2.118156909942627f,  2.146311759948731f, -1.673836588859558f, -1.004659295082092f},
    { 0.668844103813171f, -1.755281925201416f,  0.263708651065826f,  0.544651985168457f},
    {-0.479055941104889f,  0.242195174098015f,  1.683301687240601f,  0.669751882553101f},
    {-1.848919749259949f,  3.216520309448242f,  4.306654453277588f, -4.437918186187744f},
};
__constant__ float WGG[4][3] = {
    { 2.146311759948731f,  0.933696150779724f,  0.406179785728455f},
    {-1.755281925201416f,  1.310971140861511f, -0.979127824306488f},
    { 1.023769974708557f,  1.312750458717346f,  1.683301687240601f},
    { 0.915698945522308f, -2.015886306762695f,  4.437918186187744f},
};
__constant__ float WGA[4][2] = {
    { 0.175659596920013f,  0.076416060328484f},
    {-0.299846142530441f,  0.223946735262871f},
    { 0.243508785963058f,  0.312244206666946f},
    { 0.018418358638883f, -0.040547512471676f},
};

__global__ void wg_input_k(const float * x, float * dt, int IW, int IH, int IC, int TW, int TH) {
    const int T = TW * TH;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) T * IC) return;
    const int ic = (int) (idx / T);
    const int tile = (int) (idx - (int64_t) ic * T);
    const int ty = tile / TW, tx = tile - ty * TW;
    float d[4][4];
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++)
            d[a][b] = x[((int64_t) ic * IH + 2 * ty + a) * IW + 2 * tx + b];
    float t1[4][4], o[16];
    for (int u = 0; u < 4; u++)
        for (int v = 0; v < 4; v++) {
            float s = 0;
            for (int a = 0; a < 4; a++) s = fmaf(WGB[u][a], d[a][v], s);
            t1[u][v] = s;
        }
    for (int u = 0; u < 4; u++)
        for (int w = 0; w < 4; w++) {
            float s = 0;
            for (int b = 0; b < 4; b++) s = fmaf(t1[u][b], WGB[w][b], s);
            o[u * 4 + w] = s;
        }
    for (int p = 0; p < 16; p++) dt[((int64_t) p * IC + ic) * T + tile] = o[p];
}

__global__ void wg_output_k(const float * yt, const float * bias, float * out,
                            int T, int TW, int OC, int OW, int OH) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) T * OC) return;
    const int oc = (int) (idx / T);
    const int tile = (int) (idx - (int64_t) oc * T);
    const int ty = tile / TW, tx = tile - ty * TW;
    float m[16];
    for (int p = 0; p < 16; p++) m[p] = yt[((int64_t) p * OC + oc) * T + tile];
    float t1[2][4], Y[2][2];
    for (int i = 0; i < 2; i++)
        for (int v = 0; v < 4; v++) {
            float s = 0;
            for (int u = 0; u < 4; u++) s = fmaf(WGA[u][i], m[u * 4 + v], s);
            t1[i][v] = s;
        }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            float s = 0;
            for (int v = 0; v < 4; v++) s = fmaf(t1[i][v], WGA[v][j], s);
            Y[i][j] = s + bias[oc];
        }
    const int64_t OP = (int64_t) OW * OH;
    const int px = 2 * tx, py = 2 * ty;
    if (px < OW && py < OH) out[(int64_t) oc * OP + py * OW + px] = Y[0][0];
    if (px + 1 < OW && py < OH) out[(int64_t) oc * OP + py * OW + px + 1] = Y[0][1];
    if (px < OW && py + 1 < OH) out[(int64_t) oc * OP + (py + 1) * OW + px] = Y[1][0];
    if (px + 1 < OW && py + 1 < OH) out[(int64_t) oc * OP + (py + 1) * OW + px + 1] = Y[1][1];
}

int main() {
    const int IW = 10, IH = 9, IC = 3, OC = 2;
    const int OW = IW, OH = IH; // s1 p1
    const int OW_t = (OW + 1) / 2, OH_t = (OH + 1) / 2;
    const int IW_p = 2 * OW_t + 2, IH_p = 2 * OH_t + 2;
    const int T = OW_t * OH_t;
    printf("padded %dx%d tiles %d (%dx%d)\n", IW_p, IH_p, T, OW_t, OH_t);

    std::vector<float> hx((size_t) IW_p * IH_p * IC, 0.0f), hw((size_t) 9 * IC * OC), hb(OC);
    // image in the padded frame at offset (1,1)
    for (size_t i = 0; i < hx.size(); i++) hx[i] = std::sin(0.7f * (float) i) * 0.5f;
    for (size_t i = 0; i < hw.size(); i++) hw[i] = std::cos(0.3f * (float) i) * 0.4f;
    for (int i = 0; i < OC; i++) hb[i] = 0.1f * i;

    // W~ (16, IC, OC)
    std::vector<float> wg((size_t) 16 * IC * OC);
    const float(*G)[3] = WGG; // host copy of the constant (values duplicated here)
    const float Gh[4][3] = {
        { 2.146311759948731f,  0.933696150779724f,  0.406179785728455f},
        {-1.755281925201416f,  1.310971140861511f, -0.979127824306488f},
        { 1.023769974708557f,  1.312750458717346f,  1.683301687240601f},
        { 0.915698945522308f, -2.015886306762695f,  4.437918186187744f},
    };
    (void) G;
    for (int u = 0; u < 4; u++)
        for (int v = 0; v < 4; v++)
            for (int ic = 0; ic < IC; ic++)
                for (int oc = 0; oc < OC; oc++) {
                    float s = 0;
                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++)
                            s += Gh[u][a] * Gh[v][b] * hw[(((size_t) oc * IC + ic) * 3 + a) * 3 + b];
                    wg[((size_t) (u * 4 + v) * IC + ic) * OC + oc] = s;
                }

    // CPU direct conv (p=1, s1) for reference
    std::vector<float> ref((size_t) OW * OH * OC);
    for (int oc = 0; oc < OC; oc++)
        for (int y = 0; y < OH; y++)
            for (int x = 0; x < OW; x++) {
                float s = hb[oc];
                for (int ic = 0; ic < IC; ic++)
                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++)
                            s += hx[((size_t) ic * IH_p + y + a) * IW_p + x + b] *
                                 hw[(((size_t) oc * IC + ic) * 3 + a) * 3 + b];
                ref[((size_t) oc * OH + y) * OW + x] = s;
            }

    // GPU pipeline
    float *dx, *dt, *dwg, *dyt, *db, *dout;
    CK(cudaMalloc(&dx, hx.size() * 4));
    CK(cudaMalloc(&dt, (size_t) 16 * IC * T * 4));
    CK(cudaMalloc(&dwg, wg.size() * 4));
    CK(cudaMalloc(&dyt, (size_t) 16 * OC * T * 4));
    CK(cudaMalloc(&db, OC * 4));
    CK(cudaMalloc(&dout, (size_t) OW * OH * OC * 4));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dwg, wg.data(), wg.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), OC * 4, cudaMemcpyHostToDevice));

    wg_input_k<<<(int) (((size_t) T * IC + 255) / 256), 256>>>(dx, dt, IW_p, IH_p, IC, OW_t, OH_t);
    cublasHandle_t h; CB(cublasCreate(&h));
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    float alpha = 1, beta = 0;
    CB(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_T, T, OC, IC,
        &alpha, dt, T, (int64_t) IC * T, dwg, OC, (int64_t) IC * OC,
        &beta, dyt, T, (int64_t) OC * T, 16));
    wg_output_k<<<(int) (((size_t) T * OC + 255) / 256), 256>>>(dyt, db, dout, T, OW_t, OC, OW, OH);
    CK(cudaDeviceSynchronize());

    std::vector<float> got((size_t) OW * OH * OC);
    CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
    double mx = 0;
    for (size_t i = 0; i < got.size(); i++) mx = std::max(mx, (double) std::fabs(got[i] - ref[i]));
    printf("max |gpu - direct| = %g\n", mx);
    for (int i = 0; i < 4; i++) printf("ref %f gpu %f\n", ref[i], got[i]);
    return mx > 1e-3;
}
