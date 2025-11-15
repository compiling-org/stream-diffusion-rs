//! Train EEG model with comprehensive AI/ML capabilities

use stream_diffusion_rs::python::{PythonEnvironment, TrainingConfig, FineTuningConfig, EvaluationConfig};
use stream_diffusion_rs::eeg::EEGProcessor;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Training EEG model with Python integration");
    println!("========================================");
    
    // Create Python environment with default Python path
    let python_env = PythonEnvironment::default();
    println!("Using Python path: {}", python_env.python_path());
    
    // Check if required Python packages are available
    check_python_packages(&python_env)?;
    
    // Create training configuration
    let training_config = TrainingConfig {
        learning_rate: 1e-3,
        batch_size: 32,
        epochs: 50,
        dataset_path: "data/eeg_training".to_string(),
        output_path: "models/eeg_trained_model".to_string(),
    };
    
    println!("Training configuration:");
    println!("  Learning rate: {}", training_config.learning_rate);
    println!("  Batch size: {}", training_config.batch_size);
    println!("  Epochs: {}", training_config.epochs);
    println!("  Dataset path: {}", training_config.dataset_path);
    println!("  Output path: {}", training_config.output_path);
    
    // Create fine-tuning configuration
    let fine_tuning_config = FineTuningConfig {
        learning_rate: 1e-4,
        batch_size: 16,
        epochs: 20,
        concept_prompt: "attention classification".to_string(),
        output_path: "models/eeg_finetuned_model".to_string(),
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
            "resting state".to_string(),
            "attention task".to_string(),
            "meditation".to_string(),
        ],
        metrics: vec!["accuracy".to_string(), "f1_score".to_string()],
    };
    
    println!("\nEvaluation configuration:");
    println!("  Test prompts: {:?}", evaluation_config.test_prompts);
    println!("  Metrics: {:?}", evaluation_config.metrics);
    
    // Create EEG processor with Python integration
    let eeg_processor = EEGProcessor::new().with_python_model("models/eeg_classifier");
    println!("\nCreated EEG processor with Python integration");
    
    // Create output directories if they don't exist
    create_directories(&training_config, &fine_tuning_config)?;
    
    // Train the model
    println!("\nStarting EEG model training...");
    match eeg_processor.train_model(&training_config) {
        Ok(result) => {
            println!("EEG training completed successfully!");
            println!("Result: {:?}", result);
        }
        Err(e) => {
            println!("EEG training failed: {}", e);
            println!("This is expected if you don't have the required model files or dataset.");
        }
    }
    
    // Fine-tune the model
    println!("\nStarting EEG model fine-tuning...");
    match eeg_processor.fine_tune_model(&fine_tuning_config) {
        Ok(result) => {
            println!("EEG fine-tuning completed successfully!");
            println!("Result: {:?}", result);
        }
        Err(e) => {
            println!("EEG fine-tuning failed: {}", e);
            println!("This is expected if you don't have the required model files.");
        }
    }
    
    // Evaluate the model
    println!("\nStarting EEG model evaluation...");
    match eeg_processor.evaluate_model(&evaluation_config) {
        Ok(result) => {
            println!("EEG evaluation completed successfully!");
            println!("Result: {:?}", result);
        }
        Err(e) => {
            println!("EEG evaluation failed: {}", e);
            println!("This is expected if you don't have the required model files.");
        }
    }
    
    println!("\nEEG training script completed!");
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
    
    // Check MNE (for EEG processing)
    let script = r#"
import json
try:
    import mne
    print(json.dumps({
        'mne_available': True,
        'version': getattr(mne, '__version__', 'unknown')
    }))
except ImportError as e:
    print(json.dumps({
        'mne_available': False,
        'error': str(e)
    }))
"#;
    
    let result = python_env.execute_script(script, None)?;
    println!("MNE test result: {:?}", result);
    
    // Check SciPy (for signal processing)
    let script = r#"
import json
try:
    import scipy
    print(json.dumps({
        'scipy_available': True,
        'version': getattr(scipy, '__version__', 'unknown')
    }))
except ImportError as e:
    print(json.dumps({
        'scipy_available': False,
        'error': str(e)
    }))
"#;
    
    let result = python_env.execute_script(script, None)?;
    println!("SciPy test result: {:?}", result);
    
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