//! Web interface for Stream Diffusion RS

use axum::{
    extract::{DefaultBodyLimit, Path, Query},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<RwLock<crate::StreamDiffusionRs>>,
    pub registry: Arc<RwLock<crate::ModelRegistry>>,
    pub output_dir: std::path::PathBuf,
}

/// API response types
#[derive(Serialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    input_shapes: HashMap<String, Vec<i64>>,
    output_shapes: HashMap<String, Vec<i64>>,
}

#[derive(Serialize)]
struct GenerationRequest {
    prompt: String,
    model_name: String,
    steps: Option<usize>,
    guidance_scale: Option<f32>,
}

#[derive(Serialize)]
struct GenerationResponse {
    image_data: Vec<u8>,
    width: usize,
    height: usize,
    format: String,
}

#[derive(Deserialize)]
struct EEGUploadRequest {
    channel_names: Vec<String>,
    sampling_rate: f32,
}

#[derive(Serialize)]
struct EEGAnalysisResponse {
    features: crate::EEGFeatures,
    visualizations: Vec<String>, // URLs to generated plots
}

/// Create the web application
pub fn create_app(state: AppState) -> Router {
    Router::new()
        .route("/", get(serve_index))
        .route("/api/models", get(list_models))
        .route("/api/models/:name", get(get_model_info))
        .route("/api/generate", post(generate_image))
        .route("/api/eeg/analyze", post(analyze_eeg))
        .route("/api/training/start", post(start_training))
        .route("/api/training/status", get(get_training_status))
        .route("/files/:filename", get(serve_file))
        // .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB limit
        .with_state(state)
}

/// Serve the main web interface
async fn serve_index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// List available models
async fn list_models(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Json<ApiResponse<Vec<String>>> {
    let registry = state.registry.read().await;
    let models = registry.list_models();

    Json(ApiResponse {
        success: true,
        data: Some(models),
        error: None,
    })
}

/// Get model information
async fn get_model_info(
    axum::extract::State(state): axum::extract::State<AppState>,
    Path(model_name): Path<String>,
) -> Json<ApiResponse<ModelInfo>> {
    let registry = state.registry.read().await;

    if let Some(model) = registry.get_model(&model_name) {
        let info = ModelInfo {
            name: model_name,
            input_shapes: model.get_input_names().clone(),
            output_shapes: model.get_output_names().clone(),
        };

        Json(ApiResponse {
            success: true,
            data: Some(info),
            error: None,
        })
    } else {
        Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Model '{}' not found", model_name)),
        })
    }
}

/// Generate image from text prompt
async fn generate_image(
    axum::extract::State(state): axum::extract::State<AppState>,
    axum::extract::Json(request): axum::extract::Json<GenerationRequest>,
) -> Json<ApiResponse<GenerationResponse>> {
    let mut engine = state.engine.write().await;

    match engine.generate_image(&request.prompt, &request.model_name) {
        Ok(image_data) => {
            let response = GenerationResponse {
                image_data,
                width: 512,
                height: 512,
                format: "rgb".to_string(),
            };

            Json(ApiResponse {
                success: true,
                data: Some(response),
                error: None,
            })
        }
        Err(e) => Json(ApiResponse {
            success: false,
            data: None,
            error: Some(e.to_string()),
        }),
    }
}

/// Analyze uploaded EEG data
async fn analyze_eeg(
    axum::extract::State(state): axum::extract::State<AppState>,
    axum::extract::Json(request): axum::extract::Json<EEGUploadRequest>,
) -> Json<ApiResponse<EEGAnalysisResponse>> {
    // Create dummy EEG data for demonstration
    let eeg_data = crate::EEGData::new(
        ndarray::Array3::<f32>::zeros((request.channel_names.len(), 1000, 10)),
        request.sampling_rate,
        request.channel_names.clone(),
    );

    // Process EEG data
    let mut processor = crate::EEGProcessor::new();
    processor.add_filter("bandpass", crate::DigitalFilter::new(crate::FilterType::BandPass, 4, 1.0, 40.0));

    let alpha_power = processor.extract_band_power(&eeg_data, crate::FrequencyBand::Alpha).unwrap();
    let beta_power = processor.extract_band_power(&eeg_data, crate::FrequencyBand::Beta).unwrap();

    // Create features
    let mut band_powers = HashMap::new();
    band_powers.insert("Alpha".to_string(), alpha_power);
    band_powers.insert("Beta".to_string(), beta_power);

    let features = crate::EEGFeatures {
        band_powers,
        connectivity: ndarray::Array2::<f32>::zeros((request.channel_names.len(), request.channel_names.len())),
        complexity: vec![0.0; request.channel_names.len()],
    };

    // Generate visualizations
    let visualizer = crate::EEGVisualizer::new(&state.output_dir);
    let mut visualizations = Vec::new();

    for channel in 0..3.min(request.channel_names.len()) {
        for epoch in 0..2 {
            let filename = format!("eeg_ch{}_ep{}.png", channel, epoch);
            visualizer.plot_channel(&eeg_data, channel, epoch, &filename).unwrap();
            visualizations.push(format!("/files/{}", filename));
        }
    }

    let response = EEGAnalysisResponse {
        features,
        visualizations,
    };

    Json(ApiResponse {
        success: true,
        data: Some(response),
        error: None,
    })
}

/// Start model training
async fn start_training(
    axum::extract::State(state): axum::extract::State<AppState>,
    axum::extract::Json(config): axum::extract::Json<crate::TrainingConfig>,
) -> Json<ApiResponse<String>> {
    // This would start training in a background task
    // For now, return a placeholder response

    Json(ApiResponse {
        success: true,
        data: Some("Training started".to_string()),
        error: None,
    })
}

/// Get training status
async fn get_training_status(
    axum::extract::State(_state): axum::extract::State<AppState>,
) -> Json<ApiResponse<serde_json::Value>> {
    // Placeholder training status
    let status = serde_json::json!({
        "status": "running",
        "epoch": 5,
        "total_epochs": 20,
        "loss": 0.234,
        "accuracy": 0.89
    });

    Json(ApiResponse {
        success: true,
        data: Some(status),
        error: None,
    })
}

/// Serve static files
async fn serve_file(
    Path(filename): Path<String>,
) -> impl IntoResponse {
    let file_path = std::path::Path::new("output").join(filename);

    if file_path.exists() {
        match tokio::fs::read(&file_path).await {
            Ok(content) => {
                let mime = mime_guess::from_path(&file_path).first_or_octet_stream();
                ([("content-type", mime.to_string())], content).into_response()
            }
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Failed to read file").into_response(),
        }
    } else {
        (StatusCode::NOT_FOUND, "File not found").into_response()
    }
}

/// Start the web server
pub async fn start_server(host: &str, port: u16, state: AppState) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_app(state);

    let addr = format!("{}:{}", host, port);
    log::info!("Starting web server at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Web interface HTML template
const INDEX_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stream Diffusion RS</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .form-group { margin: 10px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }
        .error { background: #f8d7da; color: #721c24; }
        .success { background: #d4edda; color: #155724; }
        .tabs { display: flex; border-bottom: 1px solid #ddd; }
        .tab { padding: 10px 20px; cursor: pointer; border-bottom: 3px solid transparent; }
        .tab.active { border-bottom-color: #007bff; background: #f8f9fa; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        canvas { border: 1px solid #ddd; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Stream Diffusion RS</h1>
        <p>High-performance AI and ML research framework</p>

        <div class="tabs">
            <div class="tab active" onclick="showTab('generation')">Image Generation</div>
            <div class="tab" onclick="showTab('eeg')">EEG Analysis</div>
            <div class="tab" onclick="showTab('training')">Model Training</div>
            <div class="tab" onclick="showTab('models')">Model Management</div>
        </div>

        <div id="generation" class="tab-content active">
            <div class="section">
                <h2>🎨 Image Generation</h2>
                <div class="form-group">
                    <label for="prompt">Text Prompt:</label>
                    <textarea id="prompt" rows="3" placeholder="Describe the image you want to generate..."></textarea>
                </div>
                <div class="form-group">
                    <label for="model-select">Model:</label>
                    <select id="model-select"></select>
                </div>
                <button onclick="generateImage()">Generate Image</button>
                <div id="generation-result" class="result" style="display: none;"></div>
                <canvas id="image-canvas" width="512" height="512"></canvas>
            </div>
        </div>

        <div id="eeg" class="tab-content">
            <div class="section">
                <h2>🧠 EEG Analysis</h2>
                <div class="form-group">
                    <label for="eeg-file">EEG File:</label>
                    <input type="file" id="eeg-file" accept=".edf,.bdf,.csv">
                </div>
                <div class="form-group">
                    <label for="sampling-rate">Sampling Rate (Hz):</label>
                    <input type="number" id="sampling-rate" value="250">
                </div>
                <button onclick="analyzeEEG()">Analyze EEG</button>
                <div id="eeg-result" class="result" style="display: none;"></div>
                <div id="visualizations"></div>
            </div>
        </div>

        <div id="training" class="tab-content">
            <div class="section">
                <h2>🎓 Model Training</h2>
                <div class="form-group">
                    <label for="epochs">Epochs:</label>
                    <input type="number" id="epochs" value="20">
                </div>
                <div class="form-group">
                    <label for="batch-size">Batch Size:</label>
                    <input type="number" id="batch-size" value="32">
                </div>
                <div class="form-group">
                    <label for="learning-rate">Learning Rate:</label>
                    <input type="number" id="learning-rate" step="0.001" value="0.001">
                </div>
                <button onclick="startTraining()">Start Training</button>
                <button onclick="getTrainingStatus()">Check Status</button>
                <div id="training-result" class="result" style="display: none;"></div>
            </div>
        </div>

        <div id="models" class="tab-content">
            <div class="section">
                <h2>🤖 Model Management</h2>
                <button onclick="loadModels()">Refresh Models</button>
                <div id="models-list" class="result" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }

        // Load available models
        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const data = await response.json();

                if (data.success) {
                    const select = document.getElementById('model-select');
                    select.innerHTML = '';
                    data.data.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        select.appendChild(option);
                    });

                    const modelsList = document.getElementById('models-list');
                    modelsList.innerHTML = '<h3>Available Models:</h3><ul>' +
                        data.data.map(model => `<li>${model}</li>`).join('') + '</ul>';
                    modelsList.style.display = 'block';
                } else {
                    showError('models-list', data.error);
                }
            } catch (error) {
                showError('models-list', error.message);
            }
        }

        // Generate image
        async function generateImage() {
            const prompt = document.getElementById('prompt').value;
            const modelName = document.getElementById('model-select').value;

            if (!prompt || !modelName) {
                showError('generation-result', 'Please enter a prompt and select a model');
                return;
            }

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, model_name: modelName })
                });

                const data = await response.json();

                if (data.success) {
                    // Display generated image
                    const canvas = document.getElementById('image-canvas');
                    const ctx = canvas.getContext('2d');
                    const imageData = ctx.createImageData(512, 512);

                    // Convert RGB data to RGBA
                    for (let i = 0; i < data.data.image_data.length; i += 3) {
                        const pixelIndex = (i / 3) * 4;
                        imageData.data[pixelIndex] = data.data.image_data[i];     // R
                        imageData.data[pixelIndex + 1] = data.data.image_data[i + 1]; // G
                        imageData.data[pixelIndex + 2] = data.data.image_data[i + 2]; // B
                        imageData.data[pixelIndex + 3] = 255; // A
                    }

                    ctx.putImageData(imageData, 0, 0);
                    showSuccess('generation-result', 'Image generated successfully!');
                } else {
                    showError('generation-result', data.error);
                }
            } catch (error) {
                showError('generation-result', error.message);
            }
        }

        // Analyze EEG
        async function analyzeEEG() {
            const fileInput = document.getElementById('eeg-file');
            const samplingRate = document.getElementById('sampling-rate').value;

            if (!fileInput.files[0]) {
                showError('eeg-result', 'Please select an EEG file');
                return;
            }

            const formData = new FormData();
            formData.append('eeg_file', fileInput.files[0]);
            formData.append('metadata', JSON.stringify({
                channel_names: [], // Would be extracted from file
                sampling_rate: parseFloat(samplingRate)
            }));

            try {
                const response = await fetch('/api/eeg/analyze', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    const resultDiv = document.getElementById('eeg-result');
                    resultDiv.innerHTML = `
                        <h3>EEG Analysis Results:</h3>
                        <p>Features extracted: ${Object.keys(data.data.features.band_powers).length} frequency bands</p>
                        <p>Connectivity matrix: ${data.data.features.connectivity.length} channels</p>
                    `;
                    resultDiv.className = 'result success';
                    resultDiv.style.display = 'block';

                    // Display visualizations
                    const vizDiv = document.getElementById('visualizations');
                    vizDiv.innerHTML = '<h3>Visualizations:</h3>' +
                        data.data.visualizations.map(url =>
                            `<img src="${url}" style="max-width: 300px; margin: 10px;" alt="EEG visualization">`
                        ).join('');
                } else {
                    showError('eeg-result', data.error);
                }
            } catch (error) {
                showError('eeg-result', error.message);
            }
        }

        // Training functions
        async function startTraining() {
            const config = {
                epochs: parseInt(document.getElementById('epochs').value),
                batch_size: parseInt(document.getElementById('batch-size').value),
                learning_rate: parseFloat(document.getElementById('learning-rate').value),
                patience: 5,
                validation_split: 0.2,
                shuffle: true,
                save_checkpoints: true,
                checkpoint_frequency: 5
            };

            try {
                const response = await fetch('/api/training/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });

                const data = await response.json();

                if (data.success) {
                    showSuccess('training-result', data.data);
                } else {
                    showError('training-result', data.error);
                }
            } catch (error) {
                showError('training-result', error.message);
            }
        }

        async function getTrainingStatus() {
            try {
                const response = await fetch('/api/training/status');
                const data = await response.json();

                if (data.success) {
                    const status = data.data;
                    const resultDiv = document.getElementById('training-result');
                    resultDiv.innerHTML = `
                        <h3>Training Status:</h3>
                        <p>Status: ${status.status}</p>
                        <p>Epoch: ${status.epoch}/${status.total_epochs}</p>
                        <p>Loss: ${status.loss.toFixed(4)}</p>
                        <p>Accuracy: ${(status.accuracy * 100).toFixed(2)}%</p>
                    `;
                    resultDiv.className = 'result success';
                    resultDiv.style.display = 'block';
                } else {
                    showError('training-result', data.error);
                }
            } catch (error) {
                showError('training-result', error.message);
            }
        }

        // Utility functions
        function showError(elementId, message) {
            const element = document.getElementById(elementId);
            element.textContent = 'Error: ' + message;
            element.className = 'result error';
            element.style.display = 'block';
        }

        function showSuccess(elementId, message) {
            const element = document.getElementById(elementId);
            element.textContent = message;
            element.className = 'result success';
            element.style.display = 'block';
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadModels();
        });
    </script>
</body>
</html>"#;

/// Start web server with default configuration
pub async fn start_default_server() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Arc::new(RwLock::new(crate::StreamDiffusionRs::new()));
    let registry = Arc::new(RwLock::new(crate::ModelRegistry::new()?));
    let output_dir = std::path::PathBuf::from("output");

    std::fs::create_dir_all(&output_dir)?;

    let state = AppState {
        engine,
        registry,
        output_dir,
    };

    start_server("127.0.0.1", 3000, state).await
}