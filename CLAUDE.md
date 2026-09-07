# tropical-gemm — project notes for Claude

Hybrid repo: a Rust workspace plus a Python package built from it.

## Layout
- Rust workspace root `Cargo.toml` — version lives in `[workspace.package]`; members inherit via `version.workspace = true`.
- Crates: `crates/tropical-gemm` (core lib, leaf), `crates/tropical-gemm-cuda` (CUDA at runtime), `crates/tropical-gemm-metal` (Metal on macOS), `crates/tropical-gemm-python` (PyO3 `cdylib`, **not** a crates.io target — shipped to PyPI as `tropical-gemm`).
- Internal dep version constraints in root `[workspace.dependencies]` must be bumped together with `[workspace.package]` version, plus `crates/tropical-gemm-python/pyproject.toml` and its `uv.lock`.

## Releasing (IMPORTANT — mostly CI-driven)
`.github/workflows/release.yml` triggers on **GitHub Release `published`** and does the heavy lifting:
- Publishes the **`tropical-gemm` lib crate** to crates.io (NOT `tropical-gemm-cuda`).
- Builds wheels for {ubuntu, macos, windows} × py{3.9–3.12} and uploads all to PyPI.

So the release procedure is:
1. Bump version everywhere (workspace `version`, the two internal dep constraints, `Cargo.lock`, `pyproject.toml`, `uv.lock`), update the changelog, validate and merge → tag `vX.Y.Z` → push.
2. `gh release create vX.Y.Z` → CI publishes the core lib crate to crates.io + all PyPI wheels. Wait for the core crate to become available before publishing dependent backends.
3. Manually dispatch `.github/workflows/publish-backend.yml` against the release tag, once with `crate=tropical-gemm-cuda` and once with `crate=tropical-gemm-metal`. It runs `cargo publish` using the repository secret (Linux for CUDA packaging, macOS for Metal). Alternatively, publish these crates locally with configured credentials after the core dependency is available. CUDA packaging does not need a toolkit/GPU; execution does.
4. Confirm both backend workflows and the release workflow succeed, and verify crates.io, PyPI, and GitHub wheel assets.

Pitfalls:
- Do **not** `cargo publish -p tropical-gemm` manually if you'll create a GitHub release — CI does it, and a duplicate makes the CI `publish-crates` job fail (no `--skip-existing`). That red ✗ is harmless (PyPI jobs are independent) but avoidable.
- Do **not** `maturin publish` locally — it produces a single non-manylinux Linux wheel PyPI rejects. Let CI build the multi-platform wheels.
- crates.io API returns empty results without a `User-Agent` header; PyPI's JSON API lags after upload — check `https://pypi.org/simple/tropical-gemm/`.
- Python `pyproject.toml` version historically drifted behind the Rust workspace; resynced at v0.3.0 (2026-06-20). Keep them in lockstep.
