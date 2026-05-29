//! NVRTC → CUBIN compilation for the tropical-gemm kernels.
//!
//! cudarc's safe `compile_ptx*` API only extracts **PTX** (`nvrtcGetPTX`), which
//! the CUDA driver must still JIT to SASS at module load. To match the C
//! reference's offline `nvcc -arch=sm_XX` build (issue #40) and to skip the
//! driver's PTX→SASS JIT on a warm start (issue #41), we compile straight to a
//! **CUBIN** (real SASS) by driving NVRTC's program lifecycle with cudarc's
//! public `nvrtc::result` helpers plus the two `nvrtc::sys` entry points it does
//! not yet wrap (`nvrtcGetCUBIN` / `nvrtcGetCUBINSize`).
//!
//! All `unsafe` is confined to this module. `get_cubin` mirrors cudarc 0.19.7's
//! own `nvrtc::result::get_ptx` (size query, then fill) — verified against the
//! installed crate source — differing only in the `*CUBIN*` entry points.

use crate::error::Result;
use cudarc::nvrtc::result as nvrtc;
use cudarc::nvrtc::sys;
use cudarc::nvrtc::CompileError;
use std::ffi::{CStr, CString};

/// NVRTC flags — the single source of truth, reused by the on-disk cache key
/// (see `context.rs`) so the key can never drift from what was actually compiled.
///
/// `--gpu-architecture=sm_{major}{minor}` targets the device's *real* SASS arch,
/// which is what lets `nvrtcGetCUBIN` produce a loadable cubin (a virtual
/// `compute_XX` arch would only yield PTX). Fast-math is intentionally omitted:
/// cudarc's `use_fast_math` only emits `--fmad=true`, a no-op for our max/min/+
/// kernels and the lone `a*b` in MaxMul. Add `"--fmad=true"` here only if exact
/// parity with the C reference's build is wanted.
///
/// No `-O3` is passed, and none is needed: NVRTC compiles device code optimized by default
/// (`-dopt=on` is implicit unless `-G` is given). The `-O3` in the C reference's `nvcc -O3`
/// is a *host*-compiler flag and does not apply to these pure-device kernels; device-side
/// optimization (ptxas) defaults to `-O3` in both NVRTC's CUBIN path and offline `nvcc`.
pub(crate) fn compile_flags(major: i32, minor: i32) -> Vec<String> {
    vec![format!("--gpu-architecture=sm_{major}{minor}")]
}

/// The loaded NVRTC (CUDA toolkit) version as `(major, minor)`.
///
/// Folded into the on-disk cubin cache key (see `context.rs`) so a cubin emitted
/// by one toolkit is never reused by a process running a different one: a cubin's
/// SASS is tied to the toolkit that produced it, and the CUDA driver is backward-
/// but not forward-compatible (a newer-toolkit cubin fails to load on an older
/// driver). This matches how Triton/CuPy/PyTorch key their kernel caches. On the
/// (unexpected) query failure we return `(0, 0)`, which still yields a stable key.
pub(crate) fn nvrtc_version() -> (i32, i32) {
    let (mut major, mut minor) = (0i32, 0i32);
    // SAFETY: nvrtcVersion only writes two ints through the provided pointers.
    let _ = unsafe { sys::nvrtcVersion(&mut major as *mut _, &mut minor as *mut _) };
    (major, minor)
}

/// RAII wrapper that destroys the NVRTC program on drop, so it is freed on every
/// exit path — early return, error, or a future panic — mirroring cudarc's own
/// `nvrtc::safe::Program: Drop`.
struct Program(sys::nvrtcProgram);

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: `self.0` came from `create_program` and is destroyed exactly once.
        let _ = unsafe { nvrtc::destroy_program(self.0) };
    }
}

/// Compile `source` to a CUBIN (real SASS) for compute capability
/// `(major, minor)`. The returned bytes are a cubin image, ready to load via
/// [`cudarc::nvrtc::Ptx::from_binary`].
pub(crate) fn compile_cubin(source: &str, major: i32, minor: i32) -> Result<Vec<u8>> {
    let flags = compile_flags(major, minor);

    let src = CString::new(source.as_bytes()).expect("kernel source contains an interior NUL byte");
    // `prog` owns the NVRTC program; its Drop frees it on every path below.
    let prog = Program(nvrtc::create_program(src.as_c_str(), None).map_err(CompileError::CreationError)?);

    // SAFETY: `prog.0` was just created and is freed only by `Program`'s Drop.
    if let Err(nvrtc) = unsafe { nvrtc::compile_program(prog.0, &flags) } {
        let log = unsafe { nvrtc::get_program_log(prog.0) }
            .map(|chars| unsafe { CStr::from_ptr(chars.as_ptr()) }.to_owned())
            .unwrap_or_default();
        return Err(CompileError::CompileError {
            nvrtc,
            options: flags,
            log,
        }
        .into());
    }

    // SAFETY: compilation succeeded, so the cubin image is available.
    unsafe { get_cubin(prog.0) }.map_err(|nvrtc| {
        // cudarc has no GetCubin error variant; reuse the rich CompileError with an
        // honest log so a (practically unreachable) cubin-read failure is not
        // mislabeled as a PTX error.
        CompileError::CompileError {
            nvrtc,
            options: flags,
            log: CString::new("nvrtcGetCUBIN failed after a successful compile")
                .unwrap_or_default(),
        }
        .into()
    })
}

/// Extract the compiled cubin image. Mirrors `cudarc::nvrtc::result::get_ptx`
/// (query size, then fill a buffer), using the `nvrtcGetCUBIN*` entry points the
/// safe layer does not wrap.
///
/// # Safety
/// `prog` must come from [`nvrtc::create_program`], have compiled successfully,
/// and not yet be destroyed.
unsafe fn get_cubin(prog: sys::nvrtcProgram) -> std::result::Result<Vec<u8>, nvrtc::NvrtcError> {
    let mut size: usize = 0;
    sys::nvrtcGetCUBINSize(prog, &mut size as *mut usize).result()?;
    let mut buf = vec![0u8; size];
    sys::nvrtcGetCUBIN(prog, buf.as_mut_ptr() as *mut std::ffi::c_char).result()?;
    Ok(buf)
}
