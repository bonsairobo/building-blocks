#![deny(
    rust_2018_compatibility,
    rust_2018_idioms,
    nonstandard_style,
    unused,
    future_incompatible
)]
#![warn(clippy::doc_markdown)]
pub mod data_sets;
pub mod test;

#[cfg(feature = "simdnoise")]
pub mod noise;
