// C API for PP-DocLayoutV3 ggml inference.
#ifndef PPDOCLAYOUT_H
#define PPDOCLAYOUT_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(PPDL_BUILD)
#    define PPDL_API __declspec(dllexport)
#  else
#    define PPDL_API __declspec(dllimport)
#  endif
#else
#  define PPDL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PPDL_IMG_SIZE   800  /* fixed input resolution (square)         */
#define PPDL_NUM_DETS   300  /* rows in every result                    */
#define PPDL_DET_FIELDS 7    /* label,score,x1,y1,x2,y2,order per row   */
#define PPDL_MASK_HW    200  /* per-detection mask is 200x200           */

#define PPDL_BACKEND_CPU  0
#define PPDL_BACKEND_CUDA 1

typedef struct ppdl_ctx ppdl_ctx;

/* Pointers stay valid until the next ppdl_infer or ppdl_free on the same ctx. */
typedef struct {
    const float   * dets;     /* [PPDL_NUM_DETS * PPDL_DET_FIELDS], score-descending */
    int32_t         num_dets; /* always PPDL_NUM_DETS */
    const int32_t * masks;    /* [PPDL_NUM_DETS * PPDL_MASK_HW * PPDL_MASK_HW] 0/1,
                                 row i corresponds to dets row i */
    double          infer_ms; /* graph compute time of this call */
} ppdl_result;

/* Load model and build the compute graph. Returns NULL on failure
   (details on stderr). backend: PPDL_BACKEND_CPU or PPDL_BACKEND_CUDA. */
PPDL_API ppdl_ctx * ppdl_init(const char * model_path, int backend);

/* Same, with intermediate-tensor dumping enabled (see ppdl_dump). */
PPDL_API ppdl_ctx * ppdl_init_ex(const char * model_path, int backend, int enable_dumps);

/* Run inference. image_chw: float32 CHW 3x800x800, RGB, /255-normalized.
   ori_h/ori_w: original image size for box coordinate restore.
   Returns 0 on success. */
PPDL_API int ppdl_infer(ppdl_ctx * ctx, const float * image_chw,
                        int ori_h, int ori_w, ppdl_result * result);

/* Write intermediate tensors of the last infer to dir (requires
   ppdl_init_ex with enable_dumps). Returns 0 on success. */
PPDL_API int ppdl_dump(ppdl_ctx * ctx, const char * dir);

/* Write raw graph outputs of the last infer with path prefix:
   <pfx>logits.bin (300x25 f32), <pfx>boxes.bin (300x4 sigmoid cxcywh f32),
   <pfx>order.bin (300x300 f32, [j + 300*i] = k_i . q_j), <pfx>sel.bin (300 i32,
   flat q*25+c indices chosen by postprocess, rank order). Returns 0 on success. */
PPDL_API int ppdl_dump_raw(ppdl_ctx * ctx, const char * pfx);

PPDL_API void ppdl_free(ppdl_ctx * ctx);

#ifdef __cplusplus
}
#endif

#endif /* PPDOCLAYOUT_H */
