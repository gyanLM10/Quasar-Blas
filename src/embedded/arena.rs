//! Lock-free static arena allocator for embedded bare-metal targets.
//!
//! Provides deterministic, zero-allocation memory management using a
//! pre-allocated static byte buffer. No `malloc`, no heap, no OS.
//!
//! ## Design
//!
//! Uses a bump allocator pattern:
//! - `alloc<T>(count)` advances an atomic pointer into the static buffer
//! - `reset()` resets the pointer to zero (frees all allocations instantly)
//! - Alignment is handled automatically based on `align_of::<T>()`
//!
//! ## Thread Safety
//!
//! The bump pointer uses `AtomicUsize` with `Ordering::Relaxed` for lock-free
//! advancement. For single-threaded embedded targets (the primary use case),
//! this has zero overhead. For multi-threaded scenarios, `Ordering::AcqRel`
//! would be needed (not implemented here as Cortex-M targets are single-core).

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

/// A fixed-size static arena allocator.
///
/// # Type Parameter
/// - `N`: The total size of the arena in bytes. Must be > 0.
///
/// # Example
/// ```ignore
/// static ARENA: StaticArena<8192> = StaticArena::new();
///
/// let buf = ARENA.alloc::<f32>(100).unwrap(); // 400 bytes
/// // ... use buf ...
/// ARENA.reset(); // Free everything
/// ```
pub struct StaticArena<const N: usize> {
    /// The raw byte buffer.
    buffer: UnsafeCell<[u8; N]>,
    /// Current allocation offset (bump pointer).
    offset: AtomicUsize,
}

// SAFETY: The arena uses atomic operations for the bump pointer and
// UnsafeCell for the buffer. On single-threaded embedded targets,
// this is trivially safe. On multi-threaded targets, the atomic
// ordering would need upgrading to AcqRel.
unsafe impl<const N: usize> Sync for StaticArena<N> {}
unsafe impl<const N: usize> Send for StaticArena<N> {}

impl<const N: usize> StaticArena<N> {
    /// Create a new zero-initialized arena.
    pub const fn new() -> Self {
        Self {
            buffer: UnsafeCell::new([0u8; N]),
            offset: AtomicUsize::new(0),
        }
    }

    /// Allocate `count` elements of type `T` from the arena.
    ///
    /// Returns `None` if the arena doesn't have enough space.
    /// The returned slice is zero-initialized.
    ///
    /// # Alignment
    /// The allocation is aligned to `align_of::<T>()`.
    pub fn alloc<T: Copy + Default>(&self, count: usize) -> Option<&mut [T]> {
        let align = core::mem::align_of::<T>();
        let size = core::mem::size_of::<T>() * count;

        if size == 0 {
            return Some(&mut []);
        }

        // Atomically advance the bump pointer
        loop {
            let current = self.offset.load(Ordering::Relaxed);

            // Align up
            let aligned = (current + align - 1) & !(align - 1);
            let new_offset = aligned + size;

            if new_offset > N {
                return None; // Out of space
            }

            // Try to advance the pointer
            match self.offset.compare_exchange_weak(
                current,
                new_offset,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // SAFETY: We have exclusive access to the range [aligned, new_offset)
                    // because compare_exchange succeeded, guaranteeing no other allocation
                    // overlaps this range.
                    let ptr = unsafe {
                        let buf_ptr = self.buffer.get() as *mut u8;
                        buf_ptr.add(aligned) as *mut T
                    };

                    // Zero-initialize
                    let slice = unsafe { core::slice::from_raw_parts_mut(ptr, count) };
                    for elem in slice.iter_mut() {
                        *elem = T::default();
                    }

                    return Some(slice);
                }
                Err(_) => continue, // CAS failed, retry
            }
        }
    }

    /// Reset the arena, freeing all allocations.
    ///
    /// This is O(1) — just resets the bump pointer to zero.
    /// All previously allocated slices become invalid after this call.
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Relaxed);
    }

    /// Current number of bytes allocated.
    pub fn used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    /// Total capacity in bytes.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Remaining bytes available.
    pub fn remaining(&self) -> usize {
        N - self.used()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_alloc() {
        let arena = StaticArena::<1024>::new();

        let buf = arena.alloc::<u8>(100).unwrap();
        assert_eq!(buf.len(), 100);
        assert!(arena.used() >= 100);
    }

    #[test]
    fn test_alignment() {
        let arena = StaticArena::<1024>::new();

        // Allocate 1 byte to offset the pointer
        let _ = arena.alloc::<u8>(1).unwrap();

        // Next allocation should be aligned to 4 bytes (f32 alignment)
        let buf = arena.alloc::<f32>(4).unwrap();
        let ptr = buf.as_ptr() as usize;
        assert_eq!(ptr % 4, 0, "f32 allocation not aligned to 4 bytes");
    }

    #[test]
    fn test_out_of_space() {
        let arena = StaticArena::<32>::new();

        let _ = arena.alloc::<u8>(30).unwrap();
        let result = arena.alloc::<u8>(10); // Would exceed capacity
        assert!(result.is_none());
    }

    #[test]
    fn test_reset() {
        let arena = StaticArena::<256>::new();

        let _ = arena.alloc::<u8>(200).unwrap();
        assert!(arena.used() >= 200);

        arena.reset();
        assert_eq!(arena.used(), 0);

        // Can allocate again after reset
        let _ = arena.alloc::<u8>(200).unwrap();
        assert!(arena.used() >= 200);
    }

    #[test]
    fn test_zero_initialization() {
        let arena = StaticArena::<256>::new();

        let buf = arena.alloc::<i32>(10).unwrap();
        for &val in buf.iter() {
            assert_eq!(val, 0, "Arena allocation not zero-initialized");
        }
    }

    #[test]
    fn test_multiple_allocs() {
        let arena = StaticArena::<1024>::new();

        let a = arena.alloc::<f32>(10).unwrap();
        let b = arena.alloc::<i8>(50).unwrap();
        let c = arena.alloc::<i32>(5).unwrap();

        assert_eq!(a.len(), 10);
        assert_eq!(b.len(), 50);
        assert_eq!(c.len(), 5);

        // Write to each without overlap
        a[0] = 42.0;
        b[0] = 127;
        c[0] = -1;

        assert_eq!(a[0], 42.0);
        assert_eq!(b[0], 127);
        assert_eq!(c[0], -1);
    }
}
