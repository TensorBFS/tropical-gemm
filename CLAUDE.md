# tropical-gemm — project notes for Claude

Hybrid repo: a Rust workspace plus a Python package built from it.

## Layout
- Rust workspace root `Cargo.toml` — version lives in `[workspace.package]`; members inherit via `version.workspace = true`.
- Crates: `crates/tropical-gemm` (core lib, leaf), `crates/tropical-gemm-cuda` (depends on the core lib, needs CUDA), `crates/tropical-gemm-python` (PyO3 `cdylib`, **not** a crates.io target — shipped to PyPI as `tropical-gemm`).
- Internal dep version constraints in root `[workspace.dependencies]` must be bumped together with `[workspace.package]` version, plus `crates/tropical-gemm-python/pyproject.toml` and its `uv.lock`.

## Releasing (IMPORTANT — mostly CI-driven)
`.github/workflows/release.yml` triggers on **GitHub Release `published`** and does the heavy lifting:
- Publishes the **`tropical-gemm` lib crate** to crates.io (NOT `tropical-gemm-cuda`).
- Builds wheels for {ubuntu, macos, windows} × py{3.9–3.12} and uploads all to PyPI.

So the release procedure is:
1. Bump version everywhere (workspace `version`, the two internal dep constraints, `pyproject.toml`, `uv.lock`) → commit → tag `vX.Y.Z` → push.
2. Manually `cargo publish -p tropical-gemm-cuda` (CI deliberately skips it; publish `-p tropical-gemm` first so the dep resolves).
3. `gh release create vX.Y.Z` → CI publishes the lib crate to crates.io + all PyPI wheels.

Pitfalls:
- Do **not** `cargo publish -p tropical-gemm` manually if you'll create a GitHub release — CI does it, and a duplicate makes the CI `publish-crates` job fail (no `--skip-existing`). That red ✗ is harmless (PyPI jobs are independent) but avoidable.
- Do **not** `maturin publish` locally — it produces a single non-manylinux Linux wheel PyPI rejects. Let CI build the multi-platform wheels.
- crates.io API returns empty results without a `User-Agent` header; PyPI's JSON API lags after upload — check `https://pypi.org/simple/tropical-gemm/`.
- Python `pyproject.toml` version historically drifted behind the Rust workspace; resynced at v0.3.0 (2026-06-20). Keep them in lockstep.
