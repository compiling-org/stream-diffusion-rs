//! Model Training Example - Custom ML model training with Stream Diffusion RS

use stream_diffusion_rs::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    log::info!("Custom Model Training Example");

    // Create training configuration
    let config = TrainingConfig {
        epochs: 50,
        batch_size: 32,
        learning_rate: 0.001,
        weight_decay: 0.0001,
        patience: 10,
        validation_split: 0.2,
        shuffle: true,
        save_checkpoints: true,
        checkpoint_frequency: 10,
    };

    // Initialize trainer
    let mut trainer = ModelTrainer::new(config, Optimizer::Adam { beta1: 0.9, beta2: 0.999 }, LossFunction::MSE);

    // Generate synthetic training data
    log::info!("Generating synthetic training data...");
    let train_data = generate_synthetic_data(1000);
    let val_data = generate_synthetic_data(200);

    // Create and train a simple neural network
    let mut model = SimpleNN::new(10, 64, 1); // 10 inputs, 64 hidden, 1 output

    log::info!("Starting training...");
    trainer.train(&mut model, &train_data.features, &train_data.labels, Some(&val_data.features), Some(&val_data.labels))?;

    // Evaluate the trained model
    log::info!("Evaluating trained model...");
    let test_data = generate_synthetic_data(100);
    evaluate_model(&model, &test_data)?;

    // Save the trained model
    save_model(&model, "trained_model.bin")?;

    log::info!("Training completed successfully!");

    Ok(())
}

/// Generate synthetic regression data for demonstration
fn generate_synthetic_data(num_samples: usize) -> TrainingData {
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::{Normal, Uniform};

    let input_size = 10;
    let mut features = ndarray::Array4::<f32>::zeros((num_samples, 1, 1, input_size));
    let mut labels = ndarray::Array2::<f32>::zeros((num_samples, 1));

    // Generate features
    for i in 0..num_samples {
        for j in 0..input_size {
            features[[i, 0, 0, j]] = rand::random::<f32>() * 2.0 - 1.0; // Random between -1 and 1
        }

        // Generate labels as a function of features (simple linear relationship with noise)
        let mut target = 0.0;
        for j in 0..input_size {
            target += features[[i, 0, 0, j]] * (j as f32 + 1.0);
        }
        target += rand::random::<f32>() * 0.1; // Add noise

        labels[[i, 0]] = target;
    }

    TrainingData::new(features, labels)
}

/// Evaluate model performance
fn evaluate_model(model: &SimpleNN, test_data: &TrainingData) -> Result<(), Box<dyn std::error::Error>> {
    let mut predictions = Vec::new();

    // Make predictions on test data
    for i in 0..test_data.features.nrows() {
        let input = test_data.features.index_axis(ndarray::Axis(0), i);
        let input_2d = input.to_shape((1, test_data.features.ncols(), test_data.features.ncols(), test_data.features.ncols()))?;
        let pred = model.forward(&input_2d)?;
        predictions.push(pred[[0, 0]]);
    }

    // Calculate metrics
    let pred_array = ndarray::Array2::from_shape_vec((predictions.len(), 1), predictions)?;
    let mse = Metrics::mse(&pred_array, &test_data.labels);
    let mae = Metrics::mae(&pred_array, &test_data.labels);

    log::info!("Test Results:");
    log::info!("  MSE: {:.4}", mse);
    log::info!("  MAE: {:.4}", mae);

    Ok(())
}

/// Save trained model (placeholder implementation)
fn save_model(model: &SimpleNN, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Saving model to {}", path);
    // In practice, this would serialize the model weights
    Ok(())
}

/// Simple neural network for demonstration
pub struct SimpleNN {
    weights1: ndarray::Array2<f32>,
    biases1: ndarray::Array1<f32>,
    weights2: ndarray::Array2<f32>,
    biases2: ndarray::Array1<f32>,
}

impl SimpleNN {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;

        let normal = Normal::new(0.0, 0.1).unwrap();

        Self {
            weights1: ndarray::Array2::<f32>::random((input_size, hidden_size), normal),
            biases1: ndarray::Array1::<f32>::zeros(hidden_size),
            weights2: ndarray::Array2::<f32>::random((hidden_size, output_size), normal),
            biases2: ndarray::Array1::<f32>::zeros(output_size),
        }
    }

    pub fn forward(&self, input: &ndarray::Array4<f32>) -> Result<ndarray::Array2<f32>, Box<dyn std::error::Error>> {
        // Flatten input
        let batch_size = input.nrows();
        let flattened_size = input.ncols() * input.ncols() * input.ncols();
        let mut flattened = ndarray::Array2::<f32>::zeros((batch_size, flattened_size));

        for b in 0..batch_size {
            for i in 0..input.ncols() {
                for j in 0..input.ncols() {
                    for k in 0..input.ncols() {
                        let idx = i * input.ncols() * input.ncols() + j * input.ncols() + k;
                        flattened[[b, idx]] = input[[b, i, j, k]];
                    }
                }
            }
        }

        // Forward pass
        let hidden = (&flattened.dot(&self.weights1) + &self.biases1).mapv(|x| x.max(0.0)); // ReLU
        let output = hidden.dot(&self.weights2) + &self.biases2;

        Ok(output)
    }
}

impl TrainableModel for SimpleNN {
    fn forward(&self, input: &ndarray::Array4<f32>) -> Result<ndarray::Array2<f32>, Box<dyn std::error::Error>> {
        self.forward(input)
    }

    fn get_parameters(&self) -> Vec<&ndarray::Array2<f32>> {
        vec![&self.weights1, &self.weights2]
    }

    fn get_gradients(&self) -> Vec<&ndarray::Array2<f32>> {
        // Placeholder - in practice, this would return computed gradients
        vec![&self.weights1, &self.weights2]
    }
}