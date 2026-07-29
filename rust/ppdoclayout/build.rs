use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=PPDL_LIB_DIR");

    let lib_dir: PathBuf = match env::var("PPDL_LIB_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
            let root = manifest.join("..").join("..");
            let candidates = [
                root.join("build-cuda").join("Release"),
                root.join("build-cpu").join("Release"),
            ];
            candidates
                .iter()
                .find(|p| {
                    p.join("ppdoclayout_c.lib").exists()
                        || p.join("libppdoclayout_c.so").exists()
                        || p.join("libppdoclayout_c.dylib").exists()
                })
                .cloned()
                .expect(
                    "ppdoclayout_c library not found; build the C++ project first \
                     (cmake --build build-cpu --config Release) or set PPDL_LIB_DIR",
                )
        }
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=ppdoclayout_c");
}
