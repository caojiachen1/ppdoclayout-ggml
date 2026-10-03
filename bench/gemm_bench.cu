// Microbench: FP32 SGEMM vs BF16x9-emulated FP32 vs plain BF16 on RTX 5080,
// over the actual PP-DocLayoutV3 GEMM shapes. Also checks emulation accuracy
// against plain FP32 (max |diff| on C, relative to |C| scale).
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
#define CB(x) do { auto s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS err %d @%d\n", (int) s_, __LINE__); exit(1); } } while (0)

struct shape { const char * name; int m, n, k; };

// row-major C[m,n] = A[m,k] * B[k,n]
static float time_sgemm(cublasHandle_t h, int m, int n, int k,
                        const float * A, const float * B, float * C,
                        bool emu_bf16x9, bool bf16_plain, int iters) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const float alpha = 1.0f, beta = 0.0f;
    // warmup
    if (!emu_bf16x9 && !bf16_plain) {
        CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
        CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                        B, CUDA_R_32F, k, A, CUDA_R_32F, k, &beta,
                        C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else if (bf16_plain) {
        CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                        B, CUDA_R_16BF, k, A, CUDA_R_16BF, k, &beta,
                        C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else {
        CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
        auto st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                        B, CUDA_R_32F, k, A, CUDA_R_32F, k, &beta,
                        C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
                        CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) return -1.0f;
    }
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) {
        if (!emu_bf16x9 && !bf16_plain) {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                         B, CUDA_R_32F, k, A, CUDA_R_32F, k, &beta,
                         C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        } else if (bf16_plain) {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                         B, CUDA_R_16BF, k, A, CUDA_R_16BF, k, &beta,
                         C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        } else {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                         B, CUDA_R_32F, k, A, CUDA_R_32F, k, &beta,
                         C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
                         CUBLAS_GEMM_DEFAULT);
        }
    }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / iters;
}

__global__ static void fill_rand(float * p, int n, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s;
    curand_init(seed, i, 0, &s);
    p[i] = curand_normal(&s) * 0.1f;
}

int main() {
    cublasHandle_t h;
    CB(cublasCreate(&h));
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH)); // no TF32 anywhere

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s sm_%d%d  SMs=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    shape shapes[] = {
        {"agg0.st0 k1",   40000,  64,  336},
        {"agg1.st0 k1",   40000, 128,   64},
        {"agg0.st1 k1",   10000, 256,  704},
        {"agg1.st1 k1",   10000, 512,  256},
        {"agg0.st2 k1",    2500, 512, 1664},
        {"agg1.st2 k1",    2500,1024,  512},
        {"agg0.st3 k1",     625,1024, 3328},
        {"agg1.st3 k1",     625,2048, 1024},
        {"enc_in0 k1",    10000, 256,  512},
        {"repvgg-c1/tap", 10000, 256,  256},
        {"fpn-c2 k1",     10000, 256,  512},
        {"dec-ffn",         300,2048,  256},
        {"mask-gemm",     40000, 300,   32},
    };

    const int maxm = 40000, maxn = 2048, maxk = 3328;
    float * A, * B, * C, * Cref;
    CK(cudaMalloc(&A,    (size_t) maxm * maxk * 4));
    CK(cudaMalloc(&B,    (size_t) maxk * maxn * 4));
    CK(cudaMalloc(&C,    (size_t) maxm * maxn * 4));
    CK(cudaMalloc(&Cref, (size_t) maxm * maxn * 4));
    fill_rand<<<(maxm*maxk+255)/256, 256>>>(A, maxm*maxk, 1);
    fill_rand<<<(maxk*maxn+255)/256, 256>>>(B, maxk*maxn, 2);
    CK(cudaDeviceSynchronize());

    printf("%-16s %9s %9s %9s %7s %9s\n", "shape", "sgemm", "bf16x9", "bf16x1",
           "x9win", "x9relerr");
    for (auto & s : shapes) {
        const double flops = 2.0 * s.m * s.n * s.k;
        float t1 = time_sgemm(h, s.m, s.n, s.k, A, B, Cref, false, false, 30);
        float t2 = time_sgemm(h, s.m, s.n, s.k, A, B, C,    true,  false, 30);
        float t3 = time_sgemm(h, s.m, s.n, s.k, A, B, C,    false, true,  30);
        // accuracy of bf16x9 vs fp32
        double rel = -1;
        if (t2 >= 0) {
            int mn = s.m * s.n;
            std::vector<float> hc(mn), hr(mn);
            CK(cudaMemcpy(hc.data(), C, mn*4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hr.data(), Cref, mn*4, cudaMemcpyDeviceToHost));
            double num = 0, den = 0;
            for (int i = 0; i < mn; i++) {
                num = std::max(num, (double) std::fabs(hc[i] - hr[i]));
                den += (double) hr[i] * hr[i];
            }
            rel = num / std::sqrt(den / mn);
        }
        printf("%-16s %8.3fms %8.3fms %8.3fms %6.2fx %9.2e   (%.0f/%.0f/%.0f GFLOPS)\n",
               s.name, t1, t2, t3, t1 / t2, rel,
               flops / (t1 * 1e6), t2 > 0 ? flops / (t2 * 1e6) : 0.0,
               flops / (t3 * 1e6));
    }
    return 0;
}
