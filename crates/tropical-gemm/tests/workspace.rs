use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use tropical_gemm::{
    tropical_matmul, tropical_matmul_with_argmax_with_workspace, GemmWorkspace, TropicalGemm,
    TropicalMaxPlus, TropicalMinPlus,
};

struct ObservedAllocator;
thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
fn record() {
    if TRACK.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
    }
}
unsafe impl GlobalAlloc for ObservedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record();
        System.realloc(ptr, layout, size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}
#[global_allocator]
static ALLOCATOR: ObservedAllocator = ObservedAllocator;

#[test]
fn warmed_serial_builder_has_no_allocations() {
    let mut workspace = GemmWorkspace::<f32>::new();
    let a = vec![1.; 31 * 257];
    let b = vec![2.; 257 * 17];
    let mut c = vec![TropicalMaxPlus(0.); 31 * 17];
    TropicalGemm::new(31, 17, 257).execute_with_workspace(
        &a,
        257,
        &b,
        17,
        &mut c,
        17,
        &mut workspace,
    );
    assert!(workspace.capacity_bytes() > 0);
    ALLOCATIONS.with(|n| n.set(0));
    TRACK.with(|t| t.set(true));
    for _ in 0..5 {
        TropicalGemm::new(31, 17, 257).execute_with_workspace(
            &a,
            257,
            &b,
            17,
            &mut c,
            17,
            &mut workspace,
        );
    }
    TRACK.with(|t| t.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    assert!(c.iter().all(|v| v.0 == 3.));
    workspace.clear();
    assert_eq!(workspace.capacity_bytes(), 0);
}

fn reuse(workspace: &mut GemmWorkspace<f32>) {
    for (m, n, k) in [
        (257, 33, 513),
        (7, 13, 257),
        (65, 129, 31),
        (0, 12, 2),
        (12, 0, 2),
        (6, 6, 0),
        (5, 11, 9),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 19) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 3) % 17) as f32).collect();
        let expected = tropical_matmul::<TropicalMaxPlus<f32>>(&a, m, k, &b, n);
        let mut c = vec![TropicalMaxPlus(12345.); m * (n + 3)];
        TropicalGemm::new(m, n, k).execute_with_workspace(&a, k, &b, n, &mut c, n + 3, workspace);
        for i in 0..m {
            assert_eq!(
                &c[i * (n + 3)..i * (n + 3) + n],
                &expected[i * n..(i + 1) * n]
            );
            assert!(c[i * (n + 3) + n..(i + 1) * (n + 3)]
                .iter()
                .all(|v| v.0 == 12345.));
        }
        let actual = tropical_matmul_with_argmax_with_workspace::<TropicalMinPlus<f32>>(
            &a, m, k, &b, n, workspace,
        );
        let expected =
            tropical_gemm::tropical_matmul_with_argmax::<TropicalMinPlus<f32>>(&a, m, k, &b, n);
        assert_eq!(actual.values, expected.values);
        assert_eq!(actual.argmax, expected.argmax);
    }
}

#[test]
fn workspace_reuses_storage_across_shapes_semirings_and_kernels() {
    let mut workspace = GemmWorkspace::new();
    reuse(&mut workspace);
    let capacity = workspace.capacity_bytes();
    reuse(&mut workspace);
    assert_eq!(workspace.capacity_bytes(), capacity);
    workspace.clear();
    reuse(&mut workspace);
}

#[cfg(feature = "parallel")]
#[test]
fn workspace_reuses_storage_across_thread_pools() {
    let mut workspace = GemmWorkspace::new();
    for threads in [4, 1, 2, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| reuse(&mut workspace));
    }
}
