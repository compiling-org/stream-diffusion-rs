//! Test Python integration with Stream Diffusion RS

use stream_diffusion_rs::python::{PythonEnvironment, PythonModel, TrainingConfig, FineTuningConfig, EvaluationConfig};
use stream_diffusion_rs::diffusion::DiffusionModel;
use stream_diffusion_rs::eeg::EEGProcessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Python integration with Stream Diffusion RS");
    println!("==================================================");
    
    // Create Python environment with default Python path
    let python_env = PythonEnvironment::default();
    println!("Using Python path: {}", python_env.python_path());
    
    // Test Python execution
    println!("Testing basic Python execution...");
    let result = python_env.execute_script("print('Hello from Python!'); import json; print(json.dumps({'test': 'success'}))", None)?;
    println!("Python execution result: {:?}", result);
    
    // Test simple computation
    println!("\nTesting Python computation...");
    let script = r#"
import json
import numpy as np

# Simple computation
arr = np.array([1, 2, 3, 4, 5])
result = np.sum(arr)
print(json.dumps({'sum': float(result), 'mean': float(np.mean(arr))}))
"#;
    
    let result = python_env.execute_script(script, None)?;
    println!("Computation result: {:?}", result);
    
    if let Some(data) = result.data {
        if let Some(sum_val) = data.get("sum") {
            println!("Sum from Python: {}", sum_val);
        }
        if let Some(mean_val) = data.get("mean") {
            println!("Mean from Python: {}", mean_val);
        }
    }
    
    // Test PyTorch availability
    println!("\nTesting PyTorch availability...");
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
    
    // Test diffusers availability
    println!("\nTesting diffusers availability...");
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
    
    // Test with diffusion model
    println!("\nTesting diffusion model with Python integration...");
    let diffusion_model = DiffusionModel::new().with_python_model("models/stable-diffusion-v1-4");
    println!("Created diffusion model with Python integration");
    
    // Test training config
    let training_config = TrainingConfig {
        learning_rate: 1e-4,
        batch_size: 4,
        epochs: 10,
        dataset_path: "data/training".to_string(),
        output_path: "models/trained_model".to_string(),
    };
    
    println!("Created training configuration");
    
    // Test fine-tuning config
    let fine_tuning_config = FineTuningConfig {
        learning_rate: 1e-5,
        batch_size: 1,
        epochs: 5,
        concept_prompt: "cyberpunk art style".to_string(),
        output_path: "models/finetuned_model".to_string(),
    };
    
    println!("Created fine-tuning configuration");
    
    // Test evaluation config
    let evaluation_config = EvaluationConfig {
        test_prompts: vec![
            "a beautiful landscape".to_string(),
            "a cyberpunk city".to_string(),
            "abstract art".to_string(),
        ],
        metrics: vec!["fid".to_string(), "clip_score".to_string()],
    };
    
    println!("Created evaluation configuration");
    
    // Test EEG processor with Python integration
    println!("\nTesting EEG processor with Python integration...");
    let eeg_processor = EEGProcessor::new().with_python_model("models/eeg_classifier");
    println!("Created EEG processor with Python integration");
    
    println!("\nAll tests completed successfully!");
    Ok(())
}