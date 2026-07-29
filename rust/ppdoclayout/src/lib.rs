//! Safe Rust bindings for PP-DocLayoutV3 ggml inference.
//!
//! Wraps the `ppdoclayout_c` shared library. The DLL/so (and the ggml
//! runtime libraries it depends on) must be discoverable at run time
//! (e.g. on `PATH` on Windows).
//!
//! ```no_run
//! use ppdoclayout::{Model, Backend};
//! let mut model = Model::new("model-f32.gguf", Backend::Cpu).unwrap();
//! let image = vec![0.0f32; ppdoclayout::IMAGE_ELEMS]; // CHW 3x800x800, /255
//! let out = model.infer(&image, 2339, 1654).unwrap();
//! for det in out.detections().filter(|d| d.score > 0.5) {
//!     println!("{:?}", det);
//! }
//! ```

use std::ffi::CString;
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::slice;

/// Fixed network input resolution (square).
pub const IMG_SIZE: usize = 800;
/// Number of f32 elements expected in the input image buffer (3 * 800 * 800).
pub const IMAGE_ELEMS: usize = 3 * IMG_SIZE * IMG_SIZE;
/// Rows in every inference result.
pub const NUM_DETS: usize = 300;
/// Fields per detection row: label, score, x1, y1, x2, y2, order.
pub const DET_FIELDS: usize = 7;
/// Per-detection mask side length.
pub const MASK_HW: usize = 200;
/// Pixels per mask.
pub const MASK_PIX: usize = MASK_HW * MASK_HW;

mod ffi {
    use std::os::raw::{c_char, c_int};

    #[repr(C)]
    pub struct PpdlCtx {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct PpdlResult {
        pub dets: *const f32,
        pub num_dets: i32,
        pub masks: *const i32,
        pub infer_ms: f64,
    }

    extern "C" {
        pub fn ppdl_init(model_path: *const c_char, backend: c_int) -> *mut PpdlCtx;
        pub fn ppdl_infer(
            ctx: *mut PpdlCtx,
            image_chw: *const f32,
            ori_h: c_int,
            ori_w: c_int,
            result: *mut PpdlResult,
        ) -> c_int;
        pub fn ppdl_free(ctx: *mut PpdlCtx);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
}

#[derive(Debug)]
pub enum Error {
    /// Model load / backend / graph setup failed (details on stderr).
    Init,
    /// Inference failed (details on stderr).
    Infer(i32),
    /// Input buffer has the wrong length or invalid original dimensions.
    InvalidInput(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Init => write!(f, "ppdoclayout: initialization failed"),
            Error::Infer(code) => write!(f, "ppdoclayout: inference failed (code {code})"),
            Error::InvalidInput(msg) => write!(f, "ppdoclayout: invalid input: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// One detected layout element, in original-image pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub label: i32,
    pub score: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    /// Reading-order rank of this element among all 300 queries (0 = first).
    pub order: i32,
}

/// Result of one inference. Borrows buffers owned by the [`Model`];
/// they remain valid until the next `infer` call.
pub struct Output<'a> {
    dets: &'a [f32],
    masks: &'a [i32],
    pub infer_ms: f64,
    _marker: PhantomData<&'a Model>,
}

impl<'a> Output<'a> {
    /// Detections in score-descending order (always [`NUM_DETS`] rows).
    pub fn detections(&self) -> impl Iterator<Item = Detection> + '_ {
        self.dets.chunks_exact(DET_FIELDS).map(|r| Detection {
            label: r[0] as i32,
            score: r[1],
            x1: r[2],
            y1: r[3],
            x2: r[4],
            y2: r[5],
            order: r[6] as i32,
        })
    }

    /// Raw detection matrix, `NUM_DETS x DET_FIELDS` row-major.
    pub fn raw_dets(&self) -> &[f32] {
        self.dets
    }

    /// Binary mask (0/1, 200x200 row-major) for detection row `i`.
    pub fn mask(&self, i: usize) -> &[i32] {
        &self.masks[i * MASK_PIX..(i + 1) * MASK_PIX]
    }

    /// All masks, `NUM_DETS x 200 x 200` row-major.
    pub fn raw_masks(&self) -> &[i32] {
        self.masks
    }
}

/// A loaded PP-DocLayoutV3 model bound to a backend.
pub struct Model {
    ctx: *mut ffi::PpdlCtx,
}

// The context is only mutated through &mut self.
unsafe impl Send for Model {}

impl Model {
    /// Load a GGUF model and build the compute graph on the given backend.
    pub fn new<P: AsRef<Path>>(model_path: P, backend: Backend) -> Result<Self, Error> {
        let path = CString::new(model_path.as_ref().to_string_lossy().as_bytes())
            .map_err(|_| Error::InvalidInput("model path contains NUL".into()))?;
        let be: c_int = match backend {
            Backend::Cpu => 0,
            Backend::Cuda => 1,
        };
        let ctx = unsafe { ffi::ppdl_init(path.as_ptr() as *const c_char, be) };
        if ctx.is_null() {
            return Err(Error::Init);
        }
        Ok(Model { ctx })
    }

    /// Run inference on a preprocessed image.
    ///
    /// `image_chw`: f32 CHW `3x800x800`, RGB, values divided by 255.
    /// `ori_h`/`ori_w`: original image size, used to scale boxes back.
    pub fn infer(&mut self, image_chw: &[f32], ori_h: u32, ori_w: u32) -> Result<Output<'_>, Error> {
        if image_chw.len() != IMAGE_ELEMS {
            return Err(Error::InvalidInput(format!(
                "image length {} != {}",
                image_chw.len(),
                IMAGE_ELEMS
            )));
        }
        if ori_h == 0 || ori_w == 0 {
            return Err(Error::InvalidInput("ori_h/ori_w must be > 0".into()));
        }
        let mut res = ffi::PpdlResult {
            dets: std::ptr::null(),
            num_dets: 0,
            masks: std::ptr::null(),
            infer_ms: 0.0,
        };
        let rc = unsafe {
            ffi::ppdl_infer(
                self.ctx,
                image_chw.as_ptr(),
                ori_h as c_int,
                ori_w as c_int,
                &mut res,
            )
        };
        if rc != 0 {
            return Err(Error::Infer(rc));
        }
        let n = res.num_dets as usize;
        let dets = unsafe { slice::from_raw_parts(res.dets, n * DET_FIELDS) };
        let masks = unsafe { slice::from_raw_parts(res.masks, n * MASK_PIX) };
        Ok(Output {
            dets,
            masks,
            infer_ms: res.infer_ms,
            _marker: PhantomData,
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { ffi::ppdl_free(self.ctx) };
    }
}
