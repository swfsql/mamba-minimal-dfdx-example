#![cfg_attr(target_arch = "wasm32", feature(slice_pattern))]
#![allow(clippy::erasing_op)]

pub mod common;

#[cfg(feature = "native")]
pub mod native;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use common::*;
