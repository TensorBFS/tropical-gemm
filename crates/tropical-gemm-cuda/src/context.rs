//! CUDA context and kernel management.

use crate::compile::{compile_cubin, compile_flags, nvrtc_version};
use crate::error::{CudaError, Result};
use cudarc::driver::{CudaContext as CudaCtx, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::Arc;

/// CUDA kernel source code.
const KERNEL_SOURCE: &str = include_str!("../kernels/tropical_gemm.cu");

/// Blocking parameters for f32 kernels.
pub const BLOCK_SIZE_M_F32: u32 = 64;
pub const BLOCK_SIZE_N_F32: u32 = 64;
pub const THREAD_SIZE_M: u32 = 4;
pub const THREAD_SIZE_N: u32 = 4;

/// Blocking parameters for f64 kernels.
pub const BLOCK_SIZE_M_F64: u32 = 32;
pub const BLOCK_SIZE_N_F64: u32 = 32;

/// Kernel function names.
const KERNEL_NAMES: &[&str] = &[
    // Standard GEMM kernels (f32)
    "tropical_maxplus_f32_nn",
    "tropical_minplus_f32_nn",
    "tropical_maxmul_f32_nn",
    // Standard GEMM kernels (f64)
    "tropical_maxplus_f64_nn",
    "tropical_minplus_f64_nn",
    "tropical_maxmul_f64_nn",
    // Standard GEMM kernels (i32)
    "tropical_maxplus_i32_nn",
    "tropical_minplus_i32_nn",
    "tropical_maxmul_i32_nn",
    // Standard GEMM kernels (i64)
    "tropical_maxplus_i64_nn",
    "tropical_minplus_i64_nn",
    "tropical_maxmul_i64_nn",
    // Standard GEMM kernel (bool / AndOr semiring)
    "tropical_andor_bool_nn",
    // Standard GEMM kernels (u32/u64 / Bitwise semiring)
    "tropical_bitwise_u32_nn",
    "tropical_bitwise_u64_nn",
    "tropical_andor_bool_nn_batched",
    "tropical_bitwise_u32_nn_batched",
    "tropical_bitwise_u64_nn_batched",
    // K-packed AndOr GEMM (pack contraction dim K into u32 words)
    "pack_rows_u32",
    "pack_cols_u32",
    "tropical_andor_kpack_direct_u32",
    // GEMM with argmax kernels (f32)
    "tropical_maxplus_f32_nn_with_argmax",
    "tropical_minplus_f32_nn_with_argmax",
    "tropical_maxmul_f32_nn_with_argmax",
    // GEMM with argmax kernels (f64)
    "tropical_maxplus_f64_nn_with_argmax",
    "tropical_minplus_f64_nn_with_argmax",
    "tropical_maxmul_f64_nn_with_argmax",
    // GEMM with argmax kernels (i32)
    "tropical_maxplus_i32_nn_with_argmax",
    "tropical_minplus_i32_nn_with_argmax",
    "tropical_maxmul_i32_nn_with_argmax",
    // GEMM with argmax kernels (i64)
    "tropical_maxplus_i64_nn_with_argmax",
    "tropical_minplus_i64_nn_with_argmax",
    "tropical_maxmul_i64_nn_with_argmax",
    // Backward pass kernels (gradient computation, float/double only)
    "tropical_backward_a_f32",
    "tropical_backward_b_f32",
    "tropical_backward_a_f64",
    "tropical_backward_b_f64",
    // Batched GEMM with argmax kernels (f32 only)
    "tropical_maxplus_f32_nn_batched_with_argmax",
    "tropical_minplus_f32_nn_batched_with_argmax",
    "tropical_maxmul_f32_nn_batched_with_argmax",
    // Forward batched GEMM kernels (no argmax): one launch, blockIdx.z = batch.
    "tropical_maxplus_f32_nn_batched",
    "tropical_minplus_f32_nn_batched",
    "tropical_maxmul_f32_nn_batched",
    "tropical_maxplus_f64_nn_batched",
    "tropical_minplus_f64_nn_batched",
    "tropical_maxmul_f64_nn_batched",
    "tropical_maxplus_i32_nn_batched",
    "tropical_minplus_i32_nn_batched",
    "tropical_maxmul_i32_nn_batched",
    "tropical_maxplus_i64_nn_batched",
    "tropical_minplus_i64_nn_batched",
    "tropical_maxmul_i64_nn_batched",
];

/// CUDA context for tropical GEMM operations.
///
/// Manages device selection, kernel compilation, and caching.
pub struct CudaContext {
    ctx: Arc<CudaCtx>,
    stream: Arc<CudaStream>,
    // The loaded module is kept alive for the lifetime of the context. Each
    // cached `CudaFunction` already holds an `Arc<CudaModule>`, but we retain
    // the module explicitly to keep ownership obvious.
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernels: HashMap<&'static str, CudaFunction>,
    /// Whether the kernels were loaded from the on-disk CUBIN cache rather than
    /// compiled fresh this run. Surfaced via [`CudaContext::kernel_cache_hit`].
    cache_hit: bool,
}

impl CudaContext {
    /// Create a new CUDA context on the default device (device 0).
    pub fn new() -> Result<Self> {
        Self::new_on_device(0)
    }

    /// Create a new CUDA context on a specific device.
    pub fn new_on_device(device_id: usize) -> Result<Self> {
        let ctx = CudaCtx::new(device_id)?;
        Self::from_device(ctx)
    }

    /// Create a context from an existing device.
    ///
    /// Kernels are compiled to a CUBIN (real SASS) for the device's actual
    /// compute capability — matching the C reference's `nvcc -arch=sm_XX` build
    /// (issue #40) — and the cubin is cached on disk so later processes skip the
    /// multi-second NVRTC compile *and* the driver's PTX→SASS JIT (issue #41).
    ///
    /// The cache is validated before use: the bytes must look like a real cubin
    /// (ELF/fatbin magic — `cuModuleLoadData` takes no length and does not promise
    /// to reject a malformed image safely) and must load with every kernel
    /// resolving. A stale, wrong-arch, or wrong-toolkit file is deleted and
    /// recompiled; a *transient* error (out-of-memory, ECC, missing JIT) is
    /// surfaced unchanged so a perfectly good cache is never thrown away.
    pub fn from_device(ctx: Arc<CudaCtx>) -> Result<Self> {
        let (major, minor) = ctx.compute_capability()?;
        let cache_path = cubin_cache_path(major, minor);

        // Warm path: use the cached cubin if it passes a cheap structural check
        // and loads. Distinguish a bad *image* (delete + recompile) from a
        // *transient* environment failure (keep the cache, propagate the error).
        if let Ok(bytes) = std::fs::read(&cache_path) {
            if looks_like_cubin(&bytes) {
                match Self::load(ctx.clone(), Ptx::from_binary(bytes), true) {
                    Ok(this) => return Ok(this),
                    Err(e) if is_invalid_image(&e) => {
                        let _ = std::fs::remove_file(&cache_path); // self-heal
                    }
                    Err(e) => return Err(e), // transient: don't discard a good cache
                }
            } else {
                // Truncated / garbage / foreign file: never hand it to the driver.
                let _ = std::fs::remove_file(&cache_path);
            }
        }

        // Cold path: compile real SASS for this device's arch. Cache before load
        // (best-effort) so we move the bytes straight into the module with no copy;
        // if the fresh cubin somehow fails to load, the next run's load-validation
        // deletes and recompiles it.
        let cubin = compile_cubin(KERNEL_SOURCE, major, minor)?;
        let _ = write_cache_atomic(&cache_path, &cubin);
        Self::load(ctx, Ptx::from_binary(cubin), false)
    }

    /// Load a cubin/PTX image into a context and resolve every kernel into the
    /// function map. `cache_hit` records whether `image` came from disk.
    fn load(ctx: Arc<CudaCtx>, image: Ptx, cache_hit: bool) -> Result<Self> {
        let stream = ctx.default_stream();
        let module = ctx.load_module(image)?;

        // With CUDA 12 lazy module loading (default), a bad image's error can
        // surface here at `cuModuleGetFunction` rather than at `load_module`.
        // Map a genuinely-absent symbol to `KernelNotFound`, but preserve any
        // other driver error's code so `from_device` can tell an invalid image
        // from a transient failure.
        let mut kernels = HashMap::new();
        for name in KERNEL_NAMES {
            let func = module.load_function(name).map_err(|e| {
                if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_FOUND {
                    CudaError::KernelNotFound(name.to_string())
                } else {
                    CudaError::Driver(e)
                }
            })?;
            kernels.insert(*name, func);
        }

        Ok(Self {
            ctx,
            stream,
            module,
            kernels,
            cache_hit,
        })
    }

    /// Whether this context loaded its kernels from the on-disk CUBIN cache
    /// (issue #41) instead of compiling them fresh. Useful for startup benchmarks.
    pub fn kernel_cache_hit(&self) -> bool {
        self.cache_hit
    }

    /// Get the underlying CUDA context.
    pub fn context(&self) -> &Arc<CudaCtx> {
        &self.ctx
    }

    /// Get the default CUDA stream used for memory transfers and kernel launches.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Get the device ordinal this context is bound to.
    pub fn ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    /// Get a kernel function by name.
    pub fn get_kernel(&self, name: &'static str) -> Result<CudaFunction> {
        self.kernels
            .get(name)
            .cloned()
            .ok_or_else(|| CudaError::KernelNotFound(name.to_string()))
    }

    /// Get GPU device name (the real model via `cuDeviceGetName`, e.g.
    /// "NVIDIA A800-SXM4-80GB"), falling back to the ordinal if the query fails.
    pub fn device_name(&self) -> String {
        self.ctx
            .name()
            .unwrap_or_else(|_| format!("CUDA Device {}", self.ctx.ordinal()))
    }

    /// Calculate grid dimensions for a given matrix size.
    pub fn grid_dims_f32(m: usize, n: usize) -> (u32, u32, u32) {
        let grid_x = ((m as u32) + BLOCK_SIZE_M_F32 - 1) / BLOCK_SIZE_M_F32;
        let grid_y = ((n as u32) + BLOCK_SIZE_N_F32 - 1) / BLOCK_SIZE_N_F32;
        (grid_x * grid_y, 1, 1)
    }

    /// Calculate grid dimensions for f64 kernels.
    pub fn grid_dims_f64(m: usize, n: usize) -> (u32, u32, u32) {
        let grid_x = ((m as u32) + BLOCK_SIZE_M_F64 - 1) / BLOCK_SIZE_M_F64;
        let grid_y = ((n as u32) + BLOCK_SIZE_N_F64 - 1) / BLOCK_SIZE_N_F64;
        (grid_x * grid_y, 1, 1)
    }

    /// Block dimensions for f32 kernels.
    pub fn block_dims_f32() -> (u32, u32, u32) {
        let bszm = BLOCK_SIZE_M_F32 / THREAD_SIZE_M;
        let bszn = BLOCK_SIZE_N_F32 / THREAD_SIZE_N;
        (bszm, bszn, 1)
    }

    /// Block dimensions for f64 kernels.
    pub fn block_dims_f64() -> (u32, u32, u32) {
        let bszm = BLOCK_SIZE_M_F64 / THREAD_SIZE_M;
        let bszn = BLOCK_SIZE_N_F64 / THREAD_SIZE_N;
        (bszm, bszn, 1)
    }
}

/// Cheap structural check before handing bytes to `cuModuleLoadData`, which takes
/// no length and gives no documented guarantee of safely rejecting a malformed
/// image (a truncated/garbage file could otherwise drive an out-of-bounds read).
/// Accept only a recognizable cubin (ELF) or fatbin container.
fn looks_like_cubin(bytes: &[u8]) -> bool {
    // ELF: 0x7F 'E' 'L' 'F'. NVIDIA fatbin container magic: 0x50 0xED 0x55 0xBA.
    bytes.starts_with(b"\x7fELF") || bytes.starts_with(&[0x50, 0xED, 0x55, 0xBA])
}

/// Whether `err` means the cached *image* is bad/stale/incompatible (delete it
/// and recompile) versus a transient/environment failure — out-of-memory, ECC,
/// missing JIT compiler — where the cache is fine and recompiling would not help.
/// Codes follow the CUDA Driver API `cuModuleLoadData`/`cuModuleGetFunction` docs.
fn is_invalid_image(err: &CudaError) -> bool {
    use cudarc::driver::sys::CUresult;
    match err {
        // A genuinely-absent kernel symbol means the cubin doesn't match our source.
        CudaError::KernelNotFound(_) => true,
        CudaError::Driver(e) => matches!(
            e.0,
            CUresult::CUDA_ERROR_INVALID_IMAGE
                | CUresult::CUDA_ERROR_INVALID_PTX
                | CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU
                | CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION
                | CUresult::CUDA_ERROR_INVALID_SOURCE
        ),
        _ => false,
    }
}

/// Disk path for the cached cubin. The filename spells out the arch and NVRTC
/// version so a different GPU arch or CUDA toolkit can never reuse the wrong file,
/// and a hash of (cache version, kernel source, compile flags, NVRTC version)
/// guards the rest.
///
/// The NVRTC version must partition the cache: a cubin's SASS is tied to the
/// toolkit that produced it, and the CUDA driver is backward- but not forward-
/// compatible, so a cubin built by a newer toolkit fails to load on an older
/// driver (this matches how Triton/CuPy/PyTorch key their kernel caches). A
/// wrong-arch/wrong-toolkit hit is thus impossible by construction; load-
/// validation in [`CudaContext::from_device`] heals anything else.
///
/// The hash is a fixed FNV-1a rather than `std`'s `DefaultHasher` (whose output
/// is explicitly not stable across Rust versions), so filenames stay reproducible
/// — a toolchain upgrade does not silently rename every cubin and force needless
/// cold recompiles.
fn cubin_cache_path(major: i32, minor: i32) -> std::path::PathBuf {
    // Bump to invalidate every cache file regardless of the hashed inputs.
    const CACHE_VERSION: u32 = 1;
    let (nv_major, nv_minor) = nvrtc_version();
    // Bind the byte arrays so `parts` doesn't borrow dropped temporaries.
    let cv = CACHE_VERSION.to_le_bytes();
    let nvmaj = nv_major.to_le_bytes();
    let nvmin = nv_minor.to_le_bytes();
    let flags = compile_flags(major, minor).join("\u{1f}");
    let parts: [&[u8]; 5] = [
        &cv,
        KERNEL_SOURCE.as_bytes(),
        flags.as_bytes(),
        &nvmaj,
        &nvmin,
    ];
    let key = stable_hash(&parts);
    cache_dir().join(format!(
        "{key:016x}_sm_{major}{minor}_nvrtc{nv_major}{nv_minor}.cubin"
    ))
}

/// A small, dependency-free, **stable** 64-bit hash (FNV-1a). Unlike `std`'s
/// `DefaultHasher`, its output is fixed across Rust versions and platforms, so
/// cache filenames derived from it are reproducible.
fn stable_hash(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a 64-bit offset basis
    for part in parts {
        for &byte in *part {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a 64-bit prime
        }
    }
    h
}

/// The on-disk cubin cache directory. CUDA only ever runs on Linux here, so
/// resolve the XDG cache dir directly rather than pull in a `dirs` dependency
/// (macOS merely type-checks this path; it is never executed there).
///
/// `$XDG_CACHE_HOME` and `$HOME/.cache` are already per-user. The last-resort
/// temp fallback (both unset, e.g. a stripped batch environment) is namespaced by
/// uid so two users on the same node never share a world-writable cubin path.
fn cache_dir() -> std::path::PathBuf {
    use std::path::PathBuf;
    const APP: &str = "tropical-gemm";
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir).join(APP);
    }
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join(".cache").join(APP);
    }
    let leaf = match current_uid() {
        Some(uid) => format!("{APP}-uid{uid}"),
        None => format!("{APP}-shared"),
    };
    std::env::temp_dir().join(leaf)
}

/// Best-effort current uid, used only to namespace the shared temp fallback.
/// On Linux `/proc/self` is owned by the running uid; elsewhere it is `None`
/// (and that fallback path is never executed on macOS anyway).
fn current_uid() -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata("/proc/self").ok().map(|m| m.uid())
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Atomically write `bytes` to `path`: write a uniquely-named temp file with
/// `O_EXCL`, fsync, then rename over the target.
///
/// The temp name carries the pid and a per-process counter, and `create_new`
/// (`O_EXCL`) makes a name collision an error rather than a silent overwrite of
/// a shared target — covering same-process concurrency and shared-filesystem pid
/// reuse across HPC array-job nodes. On any failure after the temp file is
/// created it is removed, so an interrupted write never orphans a `.tmp.*` file.
/// Caching is best-effort: a lost race or an unwritable cache dir just skips the
/// write, never failing context creation.
fn write_cache_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!(
        "tmp.{}.{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)?;

    // The temp file now exists: clean it up on any later failure.
    let write = f.write_all(bytes).and_then(|()| f.sync_all());
    drop(f); // close before rename (Windows-friendly; harmless on unix)
    if let Err(e) = write.and_then(|()| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}
