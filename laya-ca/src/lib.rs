//! Sunset lattice driven by [Laya](https://github.com/aovestdipaperino/laya-rust).
//!
//! Each pixel is a typed decision: prose about the pixel and its Moore
//! neighborhood goes in, one `#rrggbb` from the quantized spectrum comes out. The model never writes
//! free text, so the picture cannot contain an unparseable state.

mod decide;
mod evolve;
mod lattice;
#[cfg(feature = "infer")]
mod laya_decider;
mod life;
mod majority;
mod palette;

pub use decide::{Choice, DecideError, Decider, Prompt, StepConfig, NEXT_COLOR_INSTRUCTIONS};
pub use evolve::{evolve, StepReport};
pub use lattice::{Lattice, LatticeError};
#[cfg(feature = "infer")]
pub use laya_decider::LayaDecider;
pub use life::{
    centered_blinker, describe as life_describe, options as life_options, probes, score, soup,
    step as life_step, swatch_for_label, with_center, LifeScore, Probe, LIFE_INSTRUCTIONS,
};
pub use majority::MajorityDecider;
pub use palette::{colors, Swatch};
