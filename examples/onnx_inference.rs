//! Example: ONNX model loading and inference

use stream_diffusion_rs::*;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    log::info!("Starting ONNX inference example");

    // Initialize ONNX converter
    let mut converter = OnnxConverter::new()?;

    // Initialize model registry
    let mut registry = ModelRegistry::new();

    // Example model paths (these would be real ONNX files in practice)
    let model_paths = vec![
        ("resnet50", "models/resnet50.onnx"),
        ("efficientnet", "models/efficientnet.onnx"),
        ("yolov5", "models/yolov5.onnx"),
    ];

    // Load models (placeholder - in practice, these files would exist)
    for (name, path) in model_paths {
        log::info!("Loading model: {} from {}", name, path);

        // In practice, you would check if file exists first
        if Path::new(path).exists() {
            registry.register_model(name, Path::new(path))?;
            log::info!("Successfully loaded model: {}", name);
        } else {
            log::warn!("Model file not found: {} - skipping", path);
        }
    }

    // List available models
    let available_models = registry.list_models();
    log::info!("Available models: {:?}", available_models);

    // Test inference on each loaded model
    for model_name in &available_models {
        if let Some(model) = registry.get_model(model_name) {
            log::info!("Testing inference for model: {}", model_name);

            // Validate model
            let compatibility = registry.validate_model(model)?;
            log::info!("Model compatibility: {}", if compatibility.is_compatible { "✓" } else { "✗" });

            if !compatibility.is_compatible {
                log::warn!("Compatibility issues: {:?}", compatibility.issues);
            }

            if !compatibility.recommended_optimizations.is_empty() {
                log::info!("Recommended optimizations: {:?}", compatibility.recommended_optimizations);
            }

            // Test inference
            match ModelValidator::test_inference(model) {
                Ok(_) => log::info!("Inference test passed for {}", model_name),
                Err(e) => log::error!("Inference test failed for {}: {}", model_name, e),
            }

            // Benchmark model
            log::info!("Benchmarking model: {}", model_name);
            let benchmark = ModelValidator::benchmark_model(model, 10)?;
            log::info!("Benchmark results:");
            log::info!("  Average latency: {:.2}ms", benchmark.average_latency_ms);
            log::info!("  Min latency: {:.2}ms", benchmark.min_latency_ms);
            log::info!("  Max latency: {:.2}ms", benchmark.max_latency_ms);
            log::info!("  Throughput: {:.2} inferences/sec", benchmark.throughput);
            log::info!("  Total time: {:?}", benchmark.total_time);
        }
    }

    // Demonstrate batch conversion (placeholder)
    log::info!("Demonstrating batch model conversion...");

    let conversions = vec![
        ("source_model_1".to_string(), "path/to/pytorch/model1.pth".into(), "output/model1.onnx".into(), ModelFormat::PyTorch),
        ("source_model_2".to_string(), "path/to/tf/model2.pb".into(), "output/model2.onnx".into(), ModelFormat::TensorFlow),
    ];

    // This would convert models if the source files existed
    registry.batch_convert(conversions)?;

    // Create ONNX bridge for advanced operations
    let mut onnx_bridge = OnnxBridge::new();

    if let Some(model) = registry.get_model("resnet50") {
        onnx_bridge.load_onnx_model(Path::new("models/resnet50.onnx"), &converter)?;

        // Create dummy input for inference
        let input_shape = model.get_input_shapes().get("input").unwrap_or(&vec![1, 3, 224, 224]);
        let total_elements: usize = input_shape.iter().map(|&x| if x == -1 { 1 } else { x as usize }).product();

        // Create dummy RGB image data
        let dummy_data: Vec<f32> = (0..total_elements).map(|i| (i % 255) as f32 / 255.0).collect();

        let input_tensor = match input_shape.len() {
            4 => ort::Value::from_array((*input_shape[0] as usize, *input_shape[1] as usize, *input_shape[2] as usize, *input_shape[3] as usize), &dummy_data[..])?,
            _ => ort::Value::from_array((*input_shape[0] as usize,), &dummy_data[..])?,
        };

        let mut inputs = std::collections::HashMap::new();
        inputs.insert("input".to_string(), input_tensor);

        // Run inference
        let outputs = onnx_bridge.run_inference(inputs)?;
        log::info!("Inference completed. Output tensors: {}", outputs.len());

        for (name, tensor) in &outputs {
            log::info!("Output '{}' shape: {:?}", name, tensor.try_extract::<f32>()?.view().shape());
        }
    }

    // Demonstrate model optimization (placeholder)
    log::info!("Demonstrating model optimization...");

    if let Some(model) = registry.get_model("resnet50") {
        let optimized_path = Path::new("models/resnet50_optimized.onnx");
        converter.optimize_model(Path::new("models/resnet50.onnx"), optimized_path)?;

        log::info!("Model optimization completed (placeholder)");
    }

    log::info!("ONNX inference example completed successfully!");

    Ok(())
}