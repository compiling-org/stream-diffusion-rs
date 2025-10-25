//! # Stream Diffusion RS
//!
//! High-performance multimodal AI and ML research framework in Rust with
//! integrated NUWE-inspired features for standalone audiovisual processing.
//!
//! ## Features
//!
//! ### 🌈 **Multimodal AI Core**
//! - **Cross-Modal Fusion**: Text, image, audio, and biometric data integration
//! - **Multisensorial Processing**: EEG, tactile, thermal, physiological signal analysis
//! - **Real-time Streaming**: Sub-10ms latency across multiple sensory dimensions
//! - **Neuro-Emotive Intelligence**: Brain-computer interfaces with emotional context
//!
//! ### 🤖 **Diffusion Models**
//! - **UNet Architecture**: Complete implementation with attention blocks, resnet blocks, and time embeddings
//! - **DDIM Scheduler**: Advanced denoising diffusion implicit models for fast inference
//! - **Text-to-Image**: CLIP text encoding integration
//! - **Streaming Support**: Real-time image generation with progress callbacks
//! - **Multimodal Generation**: EEG-to-visual, audio-to-image cross-modal synthesis
//!
//! ### 🧠 **EEG & Neuroscience**
//! - **Signal Processing**: Filtering, frequency analysis, artifact removal
//! - **Brain Wave Analysis**: Alpha, Beta, Theta, Delta, Gamma band extraction
//! - **Real-time Processing**: Circular buffers for streaming EEG data
//! - **Neurofeedback**: Real-time brain state monitoring and feedback
//! - **Multisensorial Fusion**: EEG + biometric + environmental data integration
//! - **Cross-Modal Translation**: Brain waves to visual/audio/artistic expressions
//!
//! ### 🎛️ **Basic NUWE Features**
//! - **Simple Node Pipelines**: Lightweight node-based processing for standalone use
//! - **EEG-to-Visual Conversion**: Basic brain wave to image synthesis
//! - **Real-time Neurofeedback**: Streaming EEG analysis with visual feedback
//! - **Creative Art Generation**: EEG-influenced diffusion model outputs
//! - **Audiovisual Synchronization**: Coordinated audio-visual output generation

pub mod diffusion;
pub mod onnx;
pub mod ml;
pub mod eeg;
pub mod visualization;
pub mod training;
pub mod web;
pub mod nuwe_integration;
pub mod node_graph;
pub mod basic_nuwe_features;

// Re-export main types
pub use diffusion::*;
pub use onnx::*;
pub use ml::*;
pub use eeg::*;
pub use visualization::*;
pub use training::*;
pub use nuwe_integration::*;
pub use basic_nuwe_features::*;