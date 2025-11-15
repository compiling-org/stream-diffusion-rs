//! Train diffusion model with comprehensive AI/ML capabilities

use stream_diffusion_rs::python::{PythonEnvironment, TrainingConfig, FineTuningConfig, EvaluationConfig};
use stream_diffusion_rs::diffusion::DiffusionModel;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Training diffusion model with Python integration");
    println!("=============================================");
    
    // Create Python environment with default Python path
    let python_env = PythonEnvironment::default();
    println!("Using Python path: {}", python_env.python_path());
    
    // Check if required Python packages are available
    check_python_packages(&python_env)?;
    
    // Create training configuration
    let training_config = TrainingConfig {
        learning_rate: 1e-4,
        batch_size: 4,
        epochs: 10,
        dataset_path: "data/training".to_string(),
        output_path: "models/trained_model".to_string(),
    };
    
    println!("Training configuration:");
    println!("  Learning rate: {}", training_config.learning_rate);
    println!("  Batch size: {}", training_config.batch_size);
    println!("  Epochs: {}", training_config.epochs);
    println!("  Dataset path: {}", training_config.dataset_path);
    println!("  Output path: {}", training_config.output_path);
    
    // Create fine-tuning configuration
    let fine_tuning_config = FineTuningConfig {
        learning_rate: 1e-5,
        batch_size: 1,
        epochs: 5,
        concept_prompt: "cyberpunk art style".to_string(),
        output_path: "models/finetuned_model".to_string(),
    };
    
    println!("\nFine-tuning configuration:");
    println!("  Learning rate: {}", fine_tuning_config.learning_rate);
    println!("  Batch size: {}", fine_tuning_config.batch_size);
    println!("  Epochs: {}", fine_tuning_config.epochs);
    println!("  Concept prompt: {}", fine_tuning_config.concept_prompt);
    println!("  Output path: {}", fine_tuning_config.output_path);
    
    // Create evaluation configuration
    let evaluation_config = EvaluationConfig {
        test_prompts: vec![
            "a beautiful landscape".to_string(),
            "a cyberpunk city".to_string(),
            "abstract art".to_string(),
        ],
        metrics: vec!["fid".to_string(), "clip_score".to_string()],
    };
    
    println!("\nEvaluation configuration:");
    println!("  Test prompts: {:?}", evaluation_config.test_prompts);
    println!("  Metrics: {:?}", evaluation_config.metrics);
    
    // Create diffusion model with Python integration
    let diffusion_model = DiffusionModel::new().with_python_model("models/stable-diffusion-v1-4");
    println!("\nCreated diffusion model with Python integration");
    
    // Create output directories if they don't exist
    create_directories(&training_config, &fine_tuning_config)?;
    
    // Train the model
    println!("\nStarting model training...");
    match diffusion_model.train(&training_config) {
        Ok(result) => {
            println!("Training completed successfully!");
            println!("Result: {:?}", result);
        }
        Err(e) => {
            println!("Training failed: {}", e);
            println!("This is expected if you don't have the required model files or dataset.");
        }
    }
    
    // Fine-tune the model
    println!("\nStarting model fine-tuning...");
    match diffusion_model.fine_tune(&fine_tuning_config) {
        Ok(result) => {
            println!("Fine-tuning completed successfully!");
            println!("Result: {:?}", result);
        }
        Err(e) => {
            println!("Fine-tuning failed: {}", e);
            println!("This is expected if you don't have the required model files.");
        }
    }
    
    // Evaluate the model
    println!("\nStarting model evaluation...");
    match diffusion_model.evaluate(&evaluation_config) {
        Ok(result) => {
            println!("Evaluation completed successfully!");
            println!("Result: {:?}", result);
        }
        Err(e) => {
            println!("Evaluation failed: {}", e);
            println!("This is expected if you don't have the required model files.");
        }
    }
    
    println!("\nTraining script completed!");
    println!("Note: Actual training requires proper model files and datasets.");
    println!("This script demonstrates the integration capabilities.");
    
    Ok(())
}

fn check_python_packages(python_env: &PythonEnvironment) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nChecking Python packages...");
    
    // Check PyTorch
    let script = r#"
import json
try:
    import torch
    print(json.dumps({
        'pytorch_available': True,
        'version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }))
except ImportError as e:
    print(json.dumps({
        'pytorch_available': False,
        'error': str(e)
    }))
"#;
    
    let result = python_env.execute_script(script, None)?;
    println!("PyTorch test result: {:?}", result);
    
    // Check diffusers
    let script = r#"
import json
try:
    import diffusers
    print(json.dumps({
        'diffusers_available': True,
        'version': getattr(diffusers, '__version__', 'unknown')
    }))
except ImportError as e:
    print(json.dumps({
        'diffusers_available': False,
        'error': str(e)
    }))
"#;
    
    let result = python_env.execute_script(script, None)?;
    println!("Diffusers test result: {:?}", result);
    
    // Check transformers
    let script = r#"
import json
try:
    import transformers
    print(json.dumps({
        'transformers_available': True,
        'version': getattr(transformers, '__version__', 'unknown')
    }))
except ImportError as e:
    print(json.dumps({
        'transformers_available': False,
        'error': str(e)
    }))
"#;
    
    let result = python_env.execute_script(script, None)?;
    println!("Transformers test result: {:?}", result);
    
    Ok(())
}

fn create_directories(training_config: &TrainingConfig, fine_tuning_config: &FineTuningConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Create dataset directory
    std::fs::create_dir_all(&training_config.dataset_path)?;
    
    // Create model output directories
    std::fs::create_dir_all(Path::new(&training_config.output_path).parent().unwrap_or_else(|| Path::new(".")))?;
    std::fs::create_dir_all(Path::new(&fine_tuning_config.output_path).parent().unwrap_or_else(|| Path::new(".")))?;
    
    Ok(())
}