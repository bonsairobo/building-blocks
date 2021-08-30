use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Default, Deserialize, Serialize)]
pub struct Bitset8 {
    pub bits: u8,
}

impl Bitset8 {
    #[inline]
    pub fn bit_is_set(&self, bit: u8) -> bool {
        self.bits & (1 << bit) != 0
    }

    #[inline]
    pub fn any(&self) -> bool {
        self.bits != 0
    }

    #[inline]
    pub fn all(&self) -> bool {
        self.bits == 0xFF
    }

    #[inline]
    pub fn set_all(&mut self) {
        self.bits = 0xFF;
    }

    #[inline]
    pub fn set_bit(&mut self, bit: u8) {
        self.bits |= 1 << bit;
    }

    #[inline]
    pub fn unset_bit(&mut self, bit: u8) {
        self.bits &= !(1 << bit);
    }
}

#[derive(Default, Deserialize, Serialize)]
pub struct AtomicBitset8 {
    pub bits: AtomicU8,
}

impl Clone for AtomicBitset8 {
    fn clone(&self) -> Self {
        Self {
            bits: AtomicU8::new(self.load()),
        }
    }
}

// PERF: relax the memory ordering?
impl AtomicBitset8 {
    #[inline]
    pub fn bit_is_set(&self, bit: u8) -> bool {
        self.load() & (1 << bit) != 0
    }

    #[inline]
    pub fn any(&self) -> bool {
        self.load() != 0
    }

    #[inline]
    pub fn all(&self) -> bool {
        self.load() == 0xFF
    }

    #[inline]
    pub fn set_bit(&self, bit: u8) {
        self.bits.fetch_or(1 << bit, Ordering::SeqCst);
    }

    #[inline]
    pub fn unset_bit(&self, bit: u8) {
        self.bits.fetch_and(!(1 << bit), Ordering::SeqCst);
    }

    #[inline]
    pub fn fetch_and_unset_bit(&self, bit: u8) -> bool {
        let mask = 1 << bit;
        self.bits.fetch_and(!mask, Ordering::SeqCst) & mask != 0
    }

    #[inline]
    pub fn fetch_and_set_bit(&self, bit: u8) -> bool {
        let mask = 1 << bit;
        self.bits.fetch_or(mask, Ordering::SeqCst) & mask != 0
    }

    fn load(&self) -> u8 {
        self.bits.load(Ordering::SeqCst)
    }
}
