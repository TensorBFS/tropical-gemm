use crate::types::TropicalScalar;

/// Reusable CPU packing storage, shared by sequential calls through an exclusive borrow.
///
/// Each parallel output task gets its own pair of buffers. Storage grows to the
/// largest panels and task count used so far; call [`Self::clear`] to release it.
/// This reuses packing allocations, not result matrices or Rayon task metadata.
/// The scalar parameter permits reuse across semirings with the same scalar type.
///
/// ```
/// use tropical_gemm::{GemmWorkspace, TropicalGemm, TropicalMaxPlus};
/// let mut workspace = GemmWorkspace::<f32>::new();
/// let mut c = vec![TropicalMaxPlus(0.0); 4];
/// for _ in 0..3 {
///     TropicalGemm::new(2, 2, 2).execute_with_workspace(
///         &[1.0; 4], 2, &[2.0; 4], 2, &mut c, 2, &mut workspace,
///     );
/// }
/// assert!(c.iter().all(|v| v.0 == 3.0));
/// ```
#[derive(Debug)]
pub struct GemmWorkspace<T: TropicalScalar> {
    buffers: Vec<PackingBuffers<T>>,
}

impl<T: TropicalScalar> Default for GemmWorkspace<T> {
    fn default() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }
}

impl<T: TropicalScalar> GemmWorkspace<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes reserved by scalar packing buffers (excluding task metadata).
    pub fn capacity_bytes(&self) -> usize {
        self.buffers
            .iter()
            .map(|b| {
                b.a.capacity() * std::mem::size_of::<T>()
                    + b.b.capacity() * std::mem::size_of::<T>()
            })
            .sum()
    }

    /// Release all packing buffers. The next call will allocate them again.
    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    pub(super) fn tasks(&mut self, count: usize) -> &mut [PackingBuffers<T>] {
        self.buffers
            .resize_with(count.max(self.buffers.len()), PackingBuffers::default);
        &mut self.buffers[..count]
    }
}

#[derive(Debug)]
pub(super) struct PackingBuffers<T: TropicalScalar> {
    pub a: Vec<T>,
    pub b: Vec<T>,
}

impl<T: TropicalScalar> Default for PackingBuffers<T> {
    fn default() -> Self {
        Self {
            a: Vec::new(),
            b: Vec::new(),
        }
    }
}

impl<T: TropicalScalar> PackingBuffers<T> {
    pub fn prepare(&mut self, a_len: usize, b_len: usize) {
        // Packing overwrites every used element, including edge padding. Old
        // contents need no clearing, and unused capacity never enters a kernel.
        if self.a.len() < a_len {
            self.a.resize(a_len, T::scalar_zero());
        }
        if self.b.len() < b_len {
            self.b.resize(b_len, T::scalar_zero());
        }
    }
}
