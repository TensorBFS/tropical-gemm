# BLIS Algorithm

The CPU implementation uses BLIS-style cache blocking for optimal performance.

## 5-Loop Blocking

Matrix multiplication is blocked into tiles that fit in cache:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Loop 5: for jc in 0..N step NC    (L3 cache - columns of B)             │
│   Loop 4: for pc in 0..K step KC  (L2 cache - depth)                    │
│     Pack B[pc:KC, jc:NC] → B̃  (contiguous in L3)                        │
│     Loop 3: for ic in 0..M step MC  (L1 cache - rows of A)              │
│       Pack A[ic:MC, pc:KC] → Ã  (contiguous in L2)                      │
│       Loop 2: for jr in 0..NC step NR  (register blocking)              │
│         Loop 1: for ir in 0..MC step MR  (microkernel)                  │
│           microkernel(Ã[ir], B̃[jr], C[ic+ir, jc+jr])                    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Cache Tiling Parameters

| Parameter | Description | f32 AVX2 | f64 AVX2 | Portable |
|-----------|-------------|----------|----------|----------|
| MC | Rows per L2 block | 256 | 128 | 64 |
| NC | Columns per L3 block | 256 | 128 | 64 |
| KC | Depth per block | 512 | 256 | 256 |
| MR | Microkernel rows | 8 | 4 | 4 |
| NR | Microkernel columns | 8 | 4 | 4 |

Parameters are tuned to fit in cache:
- `MC × KC` fits in L2 cache
- `KC × NC` fits in L3 cache
- `MR × NR` fits in registers

## Packing

Before computation, matrices are **packed** into contiguous buffers:

### Pack A (MC × KC block)

Original layout (row-major):
```
A[0,0] A[0,1] A[0,2] ...
A[1,0] A[1,1] A[1,2] ...
...
```

Packed layout (MR-contiguous panels):
```
A[0,0] A[1,0] ... A[MR-1,0]   // First column of first panel
A[0,1] A[1,1] ... A[MR-1,1]   // Second column of first panel
...
A[MR,0] A[MR+1,0] ...         // First column of second panel
```

### Pack B (KC × NC block)

Packed into NR-wide panels for broadcasting:
```
B[0,0] B[0,1] ... B[0,NR-1]   // First row of first panel
B[1,0] B[1,1] ... B[1,NR-1]   // Second row of first panel
...
```

## Benefits

1. **Sequential access**: Packed data is accessed linearly
2. **Cache reuse**: Each block is loaded once, used many times
3. **TLB efficiency**: Fewer page table lookups
4. **SIMD friendly**: Contiguous data enables vectorization

## Code Location

- `core/gemm.rs`: Main blocking loops
- `core/packing.rs`: Pack functions
- `core/tiling.rs`: TilingParams struct
- `core/kernel.rs`: Microkernel trait
