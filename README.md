# Stream Diffusion RS 🧠⚡

**High-performance multimodal AI and ML research framework in Rust** - A comprehensive toolkit for diffusion models, EEG analysis, multisensorial processing, and real-time neurofeedback systems.

## 🌟 **Vision: Multimodal & Multisensorial AI**

**Combining StreamDiffusion with multimodal and multisensorial AI** describes an emerging field of high-speed, interactive generative AI that can process and create content using multiple senses at once. While StreamDiffusion focuses on the real-time processing of images and video, combining it with multimodal inputs (e.g., text, images, audio) and multisensorial AI (tactile, thermal, EEG, biometric) creates more contextually rich and responsive AI systems.

**Stream Diffusion RS** extends this vision by providing:
- **Multimodal Fusion**: Text, image, audio, and biometric data integration
- **Multisensorial Processing**: EEG, tactile, thermal, and physiological signals
- **Real-time Streaming**: Sub-10ms latency across multiple data dimensions
- **Neuro-Emotive AI**: Brain-computer interfaces with emotional intelligence
- **Cross-Modal Generation**: Converting between different sensory modalities

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

### 🌈 **Multimodal AI Core**
- **Cross-Modal Fusion**: Text, image, audio, and biometric data integration
- **Multisensorial Processing**: EEG, tactile, thermal, physiological signal analysis
- **Real-time Streaming**: Sub-10ms latency across multiple sensory dimensions
- **Neuro-Emotive Intelligence**: Brain-computer interfaces with emotional context

### 🤖 **Diffusion Models**
- **UNet Architecture**: Complete implementation with attention blocks, resnet blocks, and time embeddings
- **DDIM Scheduler**: Advanced denoising diffusion implicit models for fast inference
- **Text-to-Image**: CLIP text encoding integration
- **Streaming Support**: Real-time image generation with progress callbacks
- **Multimodal Generation**: EEG-to-visual, audio-to-image cross-modal synthesis

### 🔄 **ONNX Integration**
- **Model Conversion**: PyTorch, TensorFlow, JAX, and HuggingFace model support
- **ONNX Runtime**: High-performance inference with hardware acceleration
- **Model Registry**: Management system for multiple models
- **Burn Compatibility**: Seamless integration with Burn tensor operations

### 🧠 **EEG & Neuroscience**
- **Signal Processing**: Filtering, frequency analysis, artifact removal
- **Brain Wave Analysis**: Alpha, Beta, Theta, Delta, Gamma band extraction
- **Real-time Processing**: Circular buffers for streaming EEG data
- **Neurofeedback**: Real-time brain state monitoring and feedback
- **Multisensorial Fusion**: EEG + biometric + environmental data integration
- **Cross-Modal Translation**: Brain waves to visual/audio/artistic expressions

### 📊 **ML Research Tools**
- **Data Loading**: CSV support, batching, shuffling, cross-validation
- **Preprocessing**: Normalization, augmentation, feature scaling
- **Metrics**: MSE, MAE, accuracy, IoU, F1-score, AUC-ROC
- **Experiment Tracking**: Parameter logging, artifact storage, dashboard

### 🎨 **Visualization**
- **Plotting Engine**: Line plots, scatter plots, histograms, confusion matrices
- **EEG Visualizations**: Signal plots, topographic maps, spectrograms
- **Training Curves**: Loss and accuracy monitoring over epochs
- **Real-time Dashboard**: Live experiment monitoring

### 🎓 **Training Framework**
- **Multiple Optimizers**: SGD, Adam, RMSProp with configurable parameters
- **Loss Functions**: MSE, Cross-Entropy, Binary Cross-Entropy, Huber loss
- **Model Interface**: Trainable trait with forward/backward passes
- **Checkpointing**: Automatic model saving and loading

### 🌐 **Web Interface**
- **Gradio-like UI**: Modern tabbed interface with JavaScript
- **REST API**: Full API for image generation, EEG analysis, training
- **Interactive Features**: Real-time plotting, model management
- **File Upload**: Support for EEG data and model files

## 🏗️ **Architecture**

```
stream-diffusion-rs/
├── src/
│   ├── lib.rs              # Main library interface
│   ├── diffusion.rs        # Diffusion model implementation
│   ├── onnx.rs            # ONNX model integration
│   ├── ml.rs              # ML research utilities
│   ├── eeg.rs             # EEG analysis and processing
│   ├── visualization.rs   # Data visualization tools
│   ├── training.rs        # Model training framework
│   └── web.rs             # Web interface and API
├── examples/
│   ├── basic_usage.rs              # Basic usage example
│   ├── eeg_neurofeedback.rs        # Real-time neurofeedback
│   └── model_training_example.rs   # Custom model training
└── Cargo.toml
```

## 📦 **Installation**

Add to your `Cargo.toml`:

```toml
[dependencies]
stream-diffusion-rs = "0.1.0"
```

### Dependencies

```toml
[dependencies]
# Core ML
ndarray = "0.15"
ndarray-rand = "0.14"
rand = "0.8"

# ONNX Runtime
ort = "1.16"

# Visualization
plotters = { version = "0.3", default-features = false, features = ["svg_backend", "bitmap_backend"] }

# Web interface
axum = "0.7"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }

# Audio processing
rodio = "0.17"
hound = "3.5"

# Image processing
image = "0.24"

# Utils
anyhow = "1.0"
thiserror = "1.0"
log = "0.4"
env_logger = "0.10"
```

## 🚀 **Quick Start**

### Basic Image Generation

```rust
use stream_diffusion_rs::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Initialize the engine
    let mut engine = StreamDiffusionRs::new();

    // Load a diffusion model
    engine.load_model("stable-diffusion", "models/sd.onnx")?;

    // Configure streaming parameters
    engine.set_stream_parameters(30.0, 0.8); // 30 FPS, 80% quality

    // Generate an image
    let prompt = "A beautiful sunset over mountains, digital art style";
    let image_data = engine.generate_image(prompt, "stable-diffusion")?;

    println!("Generated image with {} bytes", image_data.len());

    Ok(())
}
```

### EEG Neurofeedback

```rust
use stream_diffusion_rs::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize EEG processor
    let mut processor = EEGProcessor::new();
    processor.add_filter("bandpass", DigitalFilter::new(FilterType::BandPass, 4, 1.0, 40.0));

    // Initialize visualizer
    let visualizer = EEGVisualizer::new(std::path::Path::new("output"));

    // Simulate real-time EEG processing
    for frame in 0..1000 {
        let eeg_data = generate_simulated_eeg_data();

        // Process EEG data
        let mut processed_data = eeg_data.clone();
        processor.remove_dc_offset(&mut processed_data);

        // Extract frequency bands
        let alpha_power = processor.extract_band_power(&processed_data, FrequencyBand::Alpha)?;
        let beta_power = processor.extract_band_power(&processed_data, FrequencyBand::Beta)?;

        // Calculate focus index
        let focus_index = calculate_focus_index(&beta_power, &alpha_power);

        // Generate feedback visualization
        visualizer.plot_eeg_topography(&values, &electrode_names, &format!("neurofeedback_{}.png", frame))?;
    }

    Ok(())
}
```

### Custom Model Training

```rust
use stream_diffusion_rs::*;

// Define your model
struct MyModel {
    weights: ndarray::Array2<f32>,
    biases: ndarray::Array1<f32>,
}

impl TrainableModel for MyModel {
    fn forward(&self, input: &ndarray::Array4<f32>) -> Result<ndarray::Array2<f32>, Box<dyn std::error::Error>> {
        // Implement forward pass
        Ok(input.sum_axis(ndarray::Axis(3)).sum_axis(ndarray::Axis(2)).sum_axis(ndarray::Axis(1)))
    }

    fn get_parameters(&self) -> Vec<&ndarray::Array2<f32>> {
        vec![&self.weights]
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create training configuration
    let config = TrainingConfig {
        epochs: 50,
        batch_size: 32,
        learning_rate: 0.001,
        ..Default::default()
    };

    // Initialize trainer
    let mut trainer = ModelTrainer::new(config, Optimizer::Adam, LossFunction::MSE);

    // Train your model
    trainer.train(&mut model, &train_features, &train_labels, Some(&val_features), Some(&val_labels))?;

    Ok(())
}
```

## 🌐 **Web Interface**

Start the web server:

```rust
use stream_diffusion_rs::web::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    start_default_server().await
}
```

Then visit `http://localhost:3000` for the interactive interface.

## 📊 **EEG Analysis Pipeline**

```rust
use stream_diffusion_rs::*;

// Load EEG data
let eeg_data = EEGData::load_from_file("eeg_data.edf")?;

// Process signals
let mut processor = EEGProcessor::new();
processor.apply_filter(&mut eeg_data, "bandpass");

// Extract features
let alpha_power = processor.extract_band_power(&eeg_data, FrequencyBand::Alpha)?;
let connectivity = processor.compute_connectivity(&eeg_data);

// Convert to audiovisual
let converter = EEGToAudiovisualConverter::new("output");
let audiovisual = converter.convert(&eeg_data)?;
```

## 🔧 **ONNX Model Integration**

```rust
use stream_diffusion_rs::*;

// Initialize converter
let converter = OnnxConverter::new()?;

// Convert PyTorch model to ONNX
converter.convert_pytorch_to_onnx(
    "model.pth",
    "model.onnx",
    &[1, 3, 224, 224],
    11
)?;

// Load and run inference
let mut bridge = OnnxBridge::new();
bridge.load_onnx_model("model.onnx", &converter)?;
bridge.convert_to_burn()?;

// Run inference
let inputs = HashMap::new();
// ... populate inputs
let outputs = bridge.run_inference_burn(&input_tensor)?;
```

## 🎯 **Performance Goals**

- **Sub-10ms latency** for neurofeedback loops
- **Real-time processing** at 250Hz EEG sampling
- **Hardware acceleration** via ONNX Runtime
- **Memory efficient** streaming processing
- **Multimodal synchronization** across sensory inputs
- **Cross-device compatibility** for VR/AR/wearables

## 🤝 **Contributing**

Contributions welcome! Areas of interest:

- **Multimodal fusion algorithms** for cross-sensory data integration
- **Additional sensor support** (thermal, tactile, biometric)
- **Burn framework integration** for GPU acceleration
- **Advanced EEG signal processing** and artifact removal
- **Real-time audio synthesis** and cross-modal generation
- **Web interface enhancements** for multisensorial visualization
- **ONNX model optimization** for edge devices
- **Neuro-emotive AI models** for emotional intelligence

## 📄 **License**

MIT License - see LICENSE file for details.

## 🙏 **Acknowledgements**

- Built with ❤️ using Rust
- Inspired by the need for high-performance ML in neuroscience
- Thanks to the Burn, ONNX Runtime, and ndarray communities