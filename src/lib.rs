//! Stream Diffusion RS - A comprehensive toolkit for diffusion models, EEG analysis, and real-time neurofeedback systems.
//!
//! This library provides a complete framework for:
//! - Diffusion model implementation with UNet architecture
//! - EEG signal processing and analysis
//! - Real-time streaming capabilities
//! - ONNX model integration
//! - Web-based interface for experimentation
//! - UI analysis and automated fixing tools

// Core modules
pub mod diffusion;
pub mod eeg;
pub mod visualization;
pub mod training;
pub mod ml;
pub mod onnx;
pub mod web;
pub mod python;
pub mod real_time_streaming;
pub mod fractal_shaders;
pub mod gesture;
pub mod synesthesia;
pub mod ai_3d_models;
pub mod bevy_integration;
#[cfg(feature = "burn-ml")]
pub mod burn;
#[cfg(feature = "tch-ml")]
pub mod tch_impl;
pub mod shader_animations;
pub mod audio_synthesis;

// UI analysis and fixing tools
pub mod ui_analyzer;
pub mod advanced_ui_analyzer;
pub mod ui_fixer;

// Re-exports for easier access
pub use diffusion::DiffusionModel;
pub use eeg::{EEGProcessor, EEGData};
pub use visualization::Plotter;
pub use training::{TrainingConfig, ModelTrainer};
pub use ml::{DataLoader, DataPreprocessor};
pub use onnx::{OnnxModel, OnnxConverter, OnnxBridge};
pub use python::{PythonEnvironment, PythonModel};
pub use real_time_streaming::RealTimeStreamingFramework;
pub use fractal_shaders::FractalShaderRenderer;
// pub use gesture::GestureSystem; // Commented out as GestureSystem doesn't exist
pub use synesthesia::SynestheticFramework;
pub use ai_3d_models::AI3DModelManager;
// #[cfg(feature = "burn-ml")]
// pub use burn::BurnModelManager;
#[cfg(feature = "tch-ml")]
pub use tch_impl::TchModelManager;
pub use shader_animations::ShaderAnimationEngine;
pub use audio_synthesis::Synthesizer;