#![warn(clippy::doc_markdown)]
pub mod data_sets;
pub mod test;

#[cfg(feature = "simdnoise")]
pub mod noise;
