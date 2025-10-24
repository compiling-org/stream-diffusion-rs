//! Example: Training a custom ML model

use stream_diffusion_rs::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    log::info!("Starting model training example");

    // Create synthetic training data
    let num_samples = 1000;
    let input_size = 28 * 28; // MNIST-like
    let num_classes = 10;

    log::info!("Generating synthetic training data: {} samples, {} features, {} classes",
               num_samples, input_size, num_classes);

    // Generate random features (simulating images)
    let normal = Normal::new(0.0, 1.0).unwrap();
    let features = ndarray::Array4::<f32>::random((num_samples, 1, 28, 28), normal);

    // Generate random labels (classification targets)
    let labels_flat: Vec<f32> = (0..num_samples)
        .map(|_| (rand::random::<f32>() * num_classes as f32) as usize as f32)
        .collect();
    let mut labels = ndarray::Array2::<f32>::zeros((num_samples, num_classes));

    for (i, &class_idx) in labels_flat.iter().enumerate() {
        labels[[i, class_idx as usize]] = 1.0; // One-hot encoding
    }

    // Create training data structure
    let mut training_data = TrainingData::new(features, labels);

    // Shuffle and split data
    training_data = training_data.shuffle();
    let (train_data, val_data) = training_data.split_train_val(0.2);

    log::info!("Training set: {} samples", train_data.features.nrows());
    log::info!("Validation set: {} samples", val_data.features.nrows());

    // Create model
    let hidden_size = 128;
    let mut model = SimpleNN::new(input_size, hidden_size, num_classes);

    log::info!("Created neural network: {} -> {} -> {}", input_size, hidden_size, num_classes);

    // Training configuration
    let config = TrainingConfig {
        epochs: 10,
        batch_size: 32,
        learning_rate: 0.001,
        weight_decay: 0.0001,
        patience: 5,
        validation_split: 0.2,
        shuffle: true,
        save_checkpoints: true,
        checkpoint_frequency: 5,
    };

    // Optimizer
    let optimizer = Optimizer::Adam {
        beta1: 0.9,
        beta2: 0.999,
    };

    // Loss function
    let loss_fn = LossFunction::CrossEntropy;

    // Create trainer
    let mut trainer = ModelTrainer::new(config, optimizer, loss_fn);

    log::info!("Starting training with configuration:");
    log::info!("  Epochs: {}", config.epochs);
    log::info!("  Batch size: {}", config.batch_size);
    log::info!("  Learning rate: {}", config.learning_rate);
    log::info!("  Optimizer: Adam");
    log::info!("  Loss: Cross-entropy");

    // Train the model
    trainer.train(
        &mut model,
        &train_data.features,
        &train_data.labels,
        Some(&val_data.features),
        Some(&val_data.labels),
    )?;

    // Get training metrics
    let metrics = trainer.get_metrics();

    log::info!("Training completed!");
    log::info!("Final training loss: {:.4}", metrics.train_loss.last().unwrap_or(&0.0));
    log::info!("Final training accuracy: {:.4}", metrics.train_accuracy.last().unwrap_or(&0.0));

    if let (Some(val_loss), Some(val_acc)) = (metrics.val_loss.last(), metrics.val_accuracy.last()) {
        log::info!("Final validation loss: {:.4}", val_loss);
        log::info!("Final validation accuracy: {:.4}", val_acc);
    }

    // Save metrics
    let output_dir = Path::new("output/training");
    std::fs::create_dir_all(output_dir)?;
    trainer.save_metrics(&output_dir.join("training_metrics.json"))?;

    log::info!("Training metrics saved to: {:?}", output_dir.join("training_metrics.json"));

    // Create visualizer and plot training curves
    let plotter = Plotter::new(output_dir);

    let epochs: Vec<f32> = (0..metrics.train_loss.len()).map(|x| x as f32).collect();

    plotter.plot_line(
        &epochs,
        &metrics.train_loss,
        "Training Loss",
        "training_loss.png"
    )?;

    if !metrics.val_loss.is_empty() {
        plotter.plot_training_curves(
            &metrics.train_loss,
            &metrics.val_loss,
            Some(&metrics.train_accuracy),
            Some(&metrics.val_accuracy),
            "training_curves.png"
        )?;
    }

    log::info!("Training curves saved to: {:?}", output_dir);

    // Test inference on a few samples
    log::info!("Testing inference on validation samples...");

    let test_predictions = model.forward(&val_data.features.slice(s![0..5, .., .., ..]))?;
    let test_labels = val_data.labels.slice(s![0..5, ..]);

    for i in 0..5 {
        let pred_class = test_predictions.row(i).iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx).unwrap_or(0);

        let true_class = test_labels.row(i).iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx).unwrap_or(0);

        log::info!("Sample {}: Predicted={}, True={}", i, pred_class, true_class);
    }

    log::info!("Model training example completed successfully!");

    Ok(())
}