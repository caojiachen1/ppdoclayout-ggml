// Run PP-DocLayoutV3 on a preprocessed input tensor and print detections.
//
// Usage:
//   cargo run --release --example detect -- <model.gguf> <input.bin> [cpu|cuda] [ori_h ori_w]
//
// input.bin: raw f32 CHW 3x800x800 (RGB, /255), e.g. ref/input.bin produced
// by test_compare.py. The ppdoclayout_c and ggml DLLs must be on PATH.

use ppdoclayout::{Backend, Model, IMAGE_ELEMS, MASK_PIX};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: detect <model.gguf> <input.bin> [cpu|cuda] [ori_h ori_w]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let input_path = &args[2];
    let backend = match args.get(3).map(|s| s.as_str()) {
        Some("cuda") => Backend::Cuda,
        _ => Backend::Cpu,
    };
    let ori_h: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(800);
    let ori_w: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(800);

    let bytes = fs::read(input_path).expect("failed to read input.bin");
    assert_eq!(bytes.len(), IMAGE_ELEMS * 4, "input.bin must be f32 3x800x800");
    let image: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let mut model = Model::new(model_path, backend).expect("model init failed");
    let out = model.infer(&image, ori_h, ori_w).expect("inference failed");

    println!("inference: {:.1} ms ({:?})", out.infer_ms, backend);
    println!("{:>5} {:>6} {:>8} {:>8} {:>8} {:>8} {:>6} {:>9}",
             "label", "score", "x1", "y1", "x2", "y2", "order", "mask_px");
    for (i, det) in out.detections().enumerate() {
        if det.score < 0.5 {
            continue;
        }
        let mask_px: i32 = out.mask(i).iter().sum();
        println!(
            "{:>5} {:>6.3} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>6} {:>8}/{}",
            det.label, det.score, det.x1, det.y1, det.x2, det.y2, det.order, mask_px, MASK_PIX
        );
    }
}
