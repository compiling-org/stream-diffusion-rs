#[cfg(feature = "burn-ml")]
use burn::tensor::Tensor;
#[cfg(feature = "burn-ml")]
use burn::nn::{Linear, LinearConfig};
#[cfg(feature = "burn-ml")]
use burn::module::Module;
#[cfg(feature = "burn-ml")]
use burn::config::Config;
#[cfg(feature = "burn-ml")]
use burn::record::{CompactRecorder, Recorder};
// Remove the duplicate import: use burn::prelude::Module;