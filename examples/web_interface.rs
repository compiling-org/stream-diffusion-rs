//! Example: Web interface for Stream Diffusion RS

use stream_diffusion_rs::web::{AppState, start_server};
use stream_diffusion_rs::onnx::ModelRegistry;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    log::info!("Starting Stream Diffusion RS Web Interface");

    // Initialize components
    // TODO: Implement StreamDiffusionRs engine
    // let engine = StreamDiffusionRs::new();
    let mut registry = ModelRegistry::new()?;
    let output_dir = std::path::PathBuf::from("output");

    std::fs::create_dir_all(&output_dir)?;

    // Load some example models (if they exist)
    let model_paths = vec![
        ("resnet50", "models/resnet50.onnx"),
        ("efficientnet", "models/efficientnet.onnx"),
    ];

    for (name, path) in model_paths {
        if std::path::Path::new(path).exists() {
            registry.register_model(name, std::path::Path::new(path))?;
            log::info!("Loaded model: {}", name);
        }
    }

    // Create app state
    // TODO: Implement StreamDiffusionRs engine
    // let state = AppState {
    //     engine: Arc::new(RwLock::new(engine)),
    //     registry: Arc::new(RwLock::new(registry)),
    //     output_dir,
    // };

    // Create app state
    // TODO: Implement StreamDiffusionRs engine
    // let state = AppState {
    //     engine: Arc::new(RwLock::new(engine)),
    //     registry: Arc::new(RwLock::new(registry)),
    //     output_dir,
    // };

    // Start web server
    log::info!("Web interface available at: http://127.0.0.1:3000");
    log::info!("Press Ctrl+C to stop the server");

    // TODO: Implement web server
    // start_server("127.0.0.1", 3000, state).await?;

    // For now, use the default server implementation
    stream_diffusion_rs::web::start_default_server().await?;

    Ok(())
}