//! Basic usage example for Stream Diffusion RS

use stream_diffusion_rs::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    log::info!("Stream Diffusion RS - Basic Usage Example");

    // Initialize the main engine
    let mut engine = StreamDiffusionRs::new();

    // Load a diffusion model (placeholder)
    engine.load_model("stable-diffusion", "models/sd.onnx")?;
    log::info!("Loaded diffusion model");

    // Configure streaming parameters
    engine.set_stream_parameters(30.0, 0.8); // 30 FPS, 80% quality

    // Generate an image
    let prompt = "A beautiful sunset over mountains, digital art style";
    match engine.generate_image(prompt, "stable-diffusion") {
        Ok(image_data) => {
            log::info!("Generated image with {} bytes", image_data.len());

            // Save image (placeholder - would save to file)
            // std::fs::write("output.png", image_data)?;
        }
        Err(e) => {
            log::error!("Failed to generate image: {}", e);
        }
    }

    // Start streaming mode
    engine.start_streaming("stable-diffusion")?;
    log::info!("Started streaming mode");

    // Simulate streaming for a few frames
    for frame in 0..10 {
        let streaming_prompt = format!("Dynamic scene frame {}", frame);
        if let Ok(_) = engine.generate_image(&streaming_prompt, "stable-diffusion") {
            log::info!("Generated streaming frame {}", frame);
        }
    }

    engine.stop_streaming();
    log::info!("Stopped streaming");

    Ok(())
}