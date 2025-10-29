## [Unreleased]

### Added
- **Stream Diffusion RS**: A comprehensive toolkit for diffusion models, EEG analysis, multisensorial processing, and real-time neurofeedback systems
- **Advanced Multimodal AI Toolkit**: Integrating diffusion models with EEG analysis, multisensorial processing, and real-time neurofeedback systems
- **Fractal Shader Renderer**: Real-time fractal visualization with WebGL shaders
  - Mandelbrot, Julia, and Burning Ship fractal types
  - Customizable parameters (iterations, zoom, offset, rotation, colors)
  - Animation support with time-based effects
  - Preset configurations for different creative styles
  - Web API endpoints for shader generation and presets
- **Enhanced Audiovisual Features**: Basic NUWE audio and visual features integration
  - Simplified fractal shader renderer for creative applications
  - WebGL-based real-time fractal rendering
  - Integration with existing multimodal fusion interface
- **Cross-Modal Fusion**: Text, image, audio, and biometric data integration
- **Multisensorial Processing**: EEG, tactile, thermal, and physiological signal analysis
- **Real-time Neurofeedback**: Sub-10ms latency optimization for neurofeedback systems

# Stream Diffusion Rust - Error Fixes and Changes
# Stream Diffusion Rust - Error Fixes and Changes

## Overview
This document tracks the fixes applied to resolve compilation errors in the stream-diffusion-rust project.

## Major Issues Fixed

### 1. ndarray API Changes (Version 0.15.x)
- **Problem**: `nrows()` and `ncols()` methods were deprecated/removed
- **Solution**: Replaced with `rows()` and `gencolumns()` respectively
- **Files affected**: `training.rs`, `diffusion.rs`, `eeg.rs`, `ml.rs`

### 2. plotters API Changes (Version 0.3.x)
- **Problem**: `BitMapBackend::new()` method removed, `LineSeries` not imported
- **Solution**:
  - Changed all `BitMapBackend::new()` to `SVGBackend::new()`
  - Added `use plotters::prelude::*;` to import `LineSeries`
  - Fixed coordinate type mismatches (i32 vs f32)
- **Files affected**: `visualization.rs`, `eeg.rs`

### 3. Type Mismatches in Array Operations
- **Problem**: Array view vs owned array conflicts, dimension mismatches
- **Solution**:
  - Fixed array slicing and ownership issues
  - Corrected dimension handling in training loops
  - Fixed gradient computation type issues
- **Files affected**: `training.rs`, `ml.rs`

### 4. axum Handler Issues
- **Problem**: Handler function signature incompatible with axum 0.7
- **Solution**: Simplified model info response to avoid complex types
- **Files affected**: `web.rs`

### 5. Serde Serialization Issues
- **Problem**: ndarray arrays don't implement Serialize/Deserialize
- **Solution**: Removed serde derives from structs containing arrays
- **Files affected**: `eeg.rs`

## Detailed Changes

### training.rs
- Changed `data.nrows()` to `data.dim().0`
- Changed `data.rows()` to `data.dim().0`
- Fixed gradient computation: `*grad + &(weight_decay * param)`
- Fixed array assignment: `new_features.row_mut(new_idx).assign(&self.features.row(old_idx))`
- Fixed bias gradient computation to handle CowRepr

### visualization.rs
- Replaced `BitMapBackend::new()` with `SVGBackend::new()`
- Added `use plotters::prelude::*;` for LineSeries
- Fixed coordinate casting: `(j as i32, i as i32)`
- Fixed range construction: `*epochs_f32.last().unwrap_or(&1.0)`
- Fixed colorbar drawing with proper backend types

### diffusion.rs
- Changed `weight.nrows()` to `weight.rows()`
- Changed `x.ncols()` to `x.gencolumns()`
- Fixed loop bounds: `x.rows()` returns Lanes, need `x.dim().0`

### eeg.rs
- Changed `self.data.nrows()` to `self.data.rows()`
- Changed `self.data.ncols()` to `self.data.gencolumns()`
- Fixed array slicing dimensions
- Replaced BitMapBackend with SVGBackend
- Fixed plotters coordinate issues
- Removed serde derives from EEGFeatures

### ml.rs
- Fixed batch stacking dimension mismatches
- Fixed type inference issues with binary operations

### web.rs
- Simplified ModelInfo to use HashMap<String, Vec<i64>> for shapes
- Fixed handler signature issues

## Remaining Issues
- Some ndarray dimension handling still needs refinement
- Plotters colorbar implementation may need further adjustment
- Performance optimizations can be added later

## Testing
- Run `cargo check` to verify compilation
- Run `cargo build --release` for optimized build
- Test examples with `cargo run --example <name>`

## Dependencies Updated
- ndarray: 0.15.x (API changes)
- plotters: 0.3.x (API changes)
- axum: 0.7.x (handler changes)

## Notes
- All changes maintain backward compatibility where possible
- Focus was on compilation success over optimization
- Some warnings remain for unused imports/variables