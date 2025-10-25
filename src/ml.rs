//! ML research utilities for data processing, metrics, and experimentation

use ndarray::{Array2, Array3, Array4, Axis};
use std::collections::HashMap;
use std::path::Path;

/// Data loader for various ML datasets
pub struct DataLoader {
    batch_size: usize,
    shuffle: bool,
    data: Vec<Array3<f32>>, // [channels, height, width] - single sample
    labels: Option<Vec<Array2<f32>>>,
    current_index: usize,
    indices: Vec<usize>,
}

impl DataLoader {
    pub fn new(batch_size: usize, shuffle: bool) -> Self {
        Self {
            batch_size,
            shuffle,
            data: Vec::new(),
            labels: None,
            current_index: 0,
            indices: Vec::new(),
        }
    }

    /// Load image dataset from directory
    pub fn load_image_dataset(&mut self, dataset_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        log::info!("Loading image dataset from: {:?}", dataset_path);

        // Scan directory for image files
        let entries = std::fs::read_dir(dataset_path)?;
        let mut image_files = Vec::new();

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    match ext.to_str().unwrap_or("").to_lowercase().as_str() {
                        "jpg" | "jpeg" | "png" | "bmp" | "tiff" => {
                            image_files.push(path);
                        }
                        _ => {}
                    }
                }
            }
        }

        log::info!("Found {} image files", image_files.len());

        // Load images (placeholder - would use image crate)
        for (i, image_path) in image_files.iter().enumerate() {
            log::debug!("Loading image {}/{}: {:?}", i + 1, image_files.len(), image_path);

            // Placeholder: create dummy image data
            // In practice, you would load actual images and preprocess them
            let dummy_image = Array3::<f32>::zeros((3, 224, 224)); // CHW format
            let dummy_label = Array2::<f32>::zeros((1, 10)); // One-hot encoded classes

            self.add_sample(dummy_image, Some(dummy_label));
        }

        self.reset();
        Ok(())
    }

    /// Load EEG dataset
    pub fn load_eeg_dataset(&mut self, dataset_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        log::info!("Loading EEG dataset from: {:?}", dataset_path);

        // Scan for EEG files
        let entries = std::fs::read_dir(dataset_path)?;
        let mut eeg_files = Vec::new();

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    match ext.to_str().unwrap_or("").to_lowercase().as_str() {
                        "edf" | "bdf" | "csv" | "mat" => {
                            eeg_files.push(path);
                        }
                        _ => {}
                    }
                }
            }
        }

        log::info!("Found {} EEG files", eeg_files.len());

        // Load EEG data (placeholder - would use proper EEG libraries)
        for (i, eeg_path) in eeg_files.iter().enumerate() {
            log::debug!("Loading EEG file {}/{}: {:?}", i + 1, eeg_files.len(), eeg_path);

            // Placeholder: create dummy EEG data
            // In practice, you would parse actual EEG files
            let dummy_eeg = Array3::<f32>::zeros((32, 1000, 1)); // [channels, time_steps, 1]
            let dummy_label = Array2::<f32>::zeros((1, 4)); // 4 classes (rest, left, right, feet)

            self.add_sample(dummy_eeg, Some(dummy_label));
        }

        self.reset();
        Ok(())
    }

    /// Add data sample
    pub fn add_sample(&mut self, data: Array3<f32>, labels: Option<Array2<f32>>) {
        self.data.push(data);
        if let Some(labels) = labels {
            if let Some(ref mut label_vec) = self.labels {
                label_vec.push(labels);
            } else {
                self.labels = Some(vec![labels]);
            }
        }
        self.reset();
    }

    /// Reset data loader for new epoch
    pub fn reset(&mut self) {
        self.current_index = 0;
        self.indices = (0..self.data.len()).collect();
        if self.shuffle {
            use ndarray_rand::RandomExt;
            use ndarray_rand::rand_distr::Uniform;
            let uniform = Uniform::new(0, self.indices.len());
            let shuffled: Vec<usize> = Array2::random((self.indices.len(), 1), uniform)
                .into_raw_vec()
                .into_iter()
                .map(|x| x as usize)
                .collect();
            self.indices = shuffled;
        }
    }

    /// Get next batch
    pub fn next_batch(&mut self) -> Option<(Array4<f32>, Option<Array2<f32>>)> {
        if self.current_index >= self.indices.len() {
            return None;
        }

        let start_idx = self.current_index;
        let end_idx = (start_idx + self.batch_size).min(self.indices.len());

        // Collect batch data
        let mut batch_data_list = Vec::new();
        let mut batch_labels_list = Vec::new();

        for &idx in &self.indices[start_idx..end_idx] {
            batch_data_list.push(self.data[idx].clone());
            if let Some(ref labels) = self.labels {
                batch_labels_list.push(labels[idx].clone());
            }
        }

        // Stack batch data - add batch dimension
        let batch_data = if batch_data_list.len() == 1 {
            // Add batch dimension: [channels, height, width] -> [1, channels, height, width]
            let sample = &batch_data_list[0];
            let (c, h, w) = sample.dim();
            let reshaped = sample.clone().into_shape((1, c, h, w)).unwrap();
            reshaped
        } else {
            // Stack along batch dimension: [channels, height, width] -> [batch, channels, height, width]
            let views: Vec<_> = batch_data_list.iter().map(|x| x.view()).collect();
            ndarray::stack(Axis(0), &views).unwrap()
        };

        let batch_labels = if !batch_labels_list.is_empty() {
            Some(if batch_labels_list.len() == 1 {
                batch_labels_list[0].clone()
            } else {
                // Stack labels along batch dimension - ensure consistent dimensions
                let views: Vec<_> = batch_labels_list.iter().map(|x| x.view()).collect();
                ndarray::stack(Axis(0), &views).unwrap().into_dimensionality::<ndarray::Ix2>().unwrap()
            })
        } else {
            None
        };

        self.current_index = end_idx;
        Some((batch_data, batch_labels))
    }

    /// Get number of batches per epoch
    pub fn num_batches(&self) -> usize {
        (self.data.len() + self.batch_size - 1) / self.batch_size
    }

    /// Get dataset statistics
    pub fn get_statistics(&self) -> DatasetStatistics {
        let mut stats = DatasetStatistics {
            num_samples: self.data.len(),
            mean: 0.0,
            std: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            shape: None,
        };

        if !self.data.is_empty() {
            let first_sample = &self.data[0];
            stats.shape = Some((1, first_sample.dim().0, first_sample.dim().1, first_sample.dim().2)); // Add batch dim

            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut count = 0usize;

            for sample in &self.data {
                for &val in sample.iter() {
                    sum += val;
                    sum_sq += val * val;
                    stats.min = stats.min.min(val);
                    stats.max = stats.max.max(val);
                    count += 1;
                }
            }

            stats.mean = sum / count as f32;
            stats.std = (sum_sq / count as f32 - stats.mean * stats.mean).sqrt();
        }

        stats
    }
}

/// Dataset statistics
#[derive(Debug, Clone)]
pub struct DatasetStatistics {
    pub num_samples: usize,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub shape: Option<(usize, usize, usize, usize)>,
}

/// Data preprocessing utilities
pub struct DataPreprocessor {
    normalizers: HashMap<String, Normalizer>,
}

impl DataPreprocessor {
    pub fn new() -> Self {
        Self {
            normalizers: HashMap::new(),
        }
    }

    /// Add normalization for a data field
    pub fn add_normalization(&mut self, field: &str, normalizer: Normalizer) {
        self.normalizers.insert(field.to_string(), normalizer);
    }

    /// Normalize data
    pub fn normalize(&self, data: Array4<f32>, field: &str) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
        if let Some(normalizer) = self.normalizers.get(field) {
            Ok(normalizer.normalize(data))
        } else {
            Ok(data) // Return unchanged if no normalizer
        }
    }

    /// Denormalize data
    pub fn denormalize(&self, data: Array4<f32>, field: &str) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
        if let Some(normalizer) = self.normalizers.get(field) {
            Ok(normalizer.denormalize(data))
        } else {
            Ok(data)
        }
    }

    /// Fit normalizers on dataset
    pub fn fit_normalizers(&mut self, data: &Array4<f32>, field: &str) {
        let normalizer = Normalizer::from_data(data);
        self.add_normalization(field, normalizer);
    }

    /// Apply random augmentations
    pub fn augment_batch(&self, data: &Array4<f32>) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
        // Placeholder augmentation pipeline
        // In practice, you would apply random rotations, flips, crops, etc.
        Ok(data.clone())
    }
}

/// Data normalization methods
#[derive(Debug, Clone)]
pub enum Normalizer {
    StandardScaler { mean: f32, std: f32 },
    MinMaxScaler { min: f32, max: f32 },
    RobustScaler { median: f32, mad: f32 },
}

impl Normalizer {
    pub fn from_data(data: &Array4<f32>) -> Self {
        let flat_data: Vec<f32> = data.iter().cloned().collect();
        let mean = flat_data.iter().sum::<f32>() / flat_data.len() as f32;
        let std = (flat_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flat_data.len() as f32).sqrt();

        Self::StandardScaler { mean, std }
    }

    pub fn normalize(&self, data: Array4<f32>) -> Array4<f32> {
        match self {
            Self::StandardScaler { mean, std } => (data - *mean) / *std,
            Self::MinMaxScaler { min, max } => (data - *min) / (max - min),
            Self::RobustScaler { median, mad } => (data - *median) / *mad,
        }
    }

    pub fn denormalize(&self, data: Array4<f32>) -> Array4<f32> {
        match self {
            Self::StandardScaler { mean, std } => data * *std + *mean,
            Self::MinMaxScaler { min, max } => data * (max - min) + *min,
            Self::RobustScaler { median, mad } => data * *mad + *median,
        }
    }
}

/// ML metrics computation
pub struct Metrics {
    values: HashMap<String, Vec<f32>>,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Add metric value
    pub fn add(&mut self, name: &str, value: f32) {
        self.values.entry(name.to_string()).or_insert(Vec::new()).push(value);
    }

    /// Get metric values
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.values.get(name).map(|v| v.as_slice())
    }

    /// Compute mean of metric
    pub fn mean(&self, name: &str) -> Option<f32> {
        self.get(name).map(|values| values.iter().sum::<f32>() / values.len() as f32)
    }

    /// Compute standard deviation of metric
    pub fn std(&self, name: &str) -> Option<f32> {
        self.mean(name).and_then(|mean| {
            self.get(name).map(|values| {
                let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
                variance.sqrt()
            })
        })
    }

    /// Compute MSE between predictions and targets
    pub fn mse(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let diff = predictions - targets;
        let squared_diff = &diff * &diff;
        squared_diff.mean().unwrap_or(0.0)
    }

    /// Compute MAE between predictions and targets
    pub fn mae(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let diff = predictions - targets;
        diff.mapv(|x| x.abs()).mean().unwrap_or(0.0)
    }

    /// Compute accuracy for classification
    pub fn accuracy(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let pred_classes = predictions.map_axis(Axis(1), |row| {
            row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        });
        let target_classes = targets.map_axis(Axis(1), |row| {
            row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        });

        let correct = pred_classes.iter().zip(target_classes.iter()).filter(|(p, t)| p == t).count();
        correct as f32 / predictions.nrows() as f32
    }

    /// Compute IoU (Intersection over Union) for segmentation
    pub fn iou(predictions: &Array3<f32>, targets: &Array3<f32>, threshold: f32) -> f32 {
        let pred_binary = predictions.mapv(|x| if x > threshold { 1.0 } else { 0.0 });
        let target_binary = targets.mapv(|x| if x > threshold { 1.0 } else { 0.0 });

        let intersection = (&pred_binary * &target_binary).sum();
        let union = (&pred_binary + &target_binary - &pred_binary * &target_binary).sum();

        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    /// Compute F1 score for binary classification
    pub fn f1_score(predictions: &Array2<f32>, targets: &Array2<f32>, threshold: f32) -> f32 {
        let pred_binary = predictions.mapv(|x| if x > threshold { 1.0 } else { 0.0 });
        let target_binary = targets.mapv(|x| if x > threshold { 1.0 } else { 0.0 });

        let true_positive = (&pred_binary * &target_binary).sum();
        let false_positive: f32 = (&pred_binary * &(1.0 - &target_binary)).sum();
        let false_negative = ((1.0 - &pred_binary) * &target_binary).sum();

        let precision = if true_positive + false_positive > 0.0 {
            true_positive / (true_positive + false_positive)
        } else {
            0.0
        };

        let recall = if true_positive + false_negative > 0.0 {
            true_positive / (true_positive + false_negative)
        } else {
            0.0
        };

        if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        }
    }

    /// Compute AUC-ROC score
    pub fn auc_roc(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        // Simplified AUC computation (placeholder)
        // In practice, you would implement proper AUC calculation
        let pred_scores: Vec<f32> = predictions.iter().cloned().collect();
        let target_labels: Vec<f32> = targets.iter().cloned().collect();

        // Sort by prediction scores
        let mut indices: Vec<usize> = (0..pred_scores.len()).collect();
        indices.sort_by(|&i, &j| pred_scores[j].partial_cmp(&pred_scores[i]).unwrap());

        let mut auc = 0.0;
        let mut prev_score = f32::NEG_INFINITY;
        let mut tp = 0.0;
        let mut fp = 0.0;
        let total_positives = target_labels.iter().sum::<f32>();
        let total_negatives = target_labels.len() as f32 - total_positives;

        for &idx in &indices {
            if pred_scores[idx] != prev_score {
                auc += tp * (fp / total_negatives);
                prev_score = pred_scores[idx];
            }

            if target_labels[idx] > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
        }

        auc += tp * (fp / total_negatives);
        auc / (total_positives * total_negatives)
    }

    /// Compute confusion matrix
    pub fn confusion_matrix(predictions: &Array2<f32>, targets: &Array2<f32>, num_classes: usize) -> Array2<f32> {
        let mut matrix = Array2::<f32>::zeros((num_classes, num_classes));

        let pred_classes = predictions.map_axis(Axis(1), |row| {
            row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        });
        let target_classes = targets.map_axis(Axis(1), |row| {
            row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        });

        for (&pred, &target) in pred_classes.iter().zip(target_classes.iter()) {
            matrix[[target, pred]] += 1.0;
        }

        matrix
    }
}

/// Experiment tracking and logging
pub struct ExperimentTracker {
    name: String,
    metrics: Metrics,
    parameters: HashMap<String, serde_json::Value>,
    artifacts: Vec<String>,
}

impl ExperimentTracker {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            metrics: Metrics::new(),
            parameters: HashMap::new(),
            artifacts: Vec::new(),
        }
    }

    /// Log parameter
    pub fn log_param(&mut self, key: &str, value: serde_json::Value) {
        self.parameters.insert(key.to_string(), value);
    }

    /// Log metric
    pub fn log_metric(&mut self, key: &str, value: f32) {
        self.metrics.add(key, value);
    }

    /// Log artifact path
    pub fn log_artifact(&mut self, path: &str) {
        self.artifacts.push(path.to_string());
    }

    /// Save experiment results
    pub fn save(&self, output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;

        fs::create_dir_all(output_dir)?;

        // Save metrics
        let metrics_path = output_dir.join("metrics.json");
        let metrics_data = serde_json::to_string_pretty(&self.metrics.values)?;
        fs::write(metrics_path, metrics_data)?;

        // Save parameters
        let params_path = output_dir.join("parameters.json");
        let params_data = serde_json::to_string_pretty(&self.parameters)?;
        fs::write(params_path, params_data)?;

        // Save artifacts list
        let artifacts_path = output_dir.join("artifacts.txt");
        let artifacts_data = self.artifacts.join("\n");
        fs::write(artifacts_path, artifacts_data)?;

        Ok(())
    }
}

/// Model checkpointing utilities
pub struct CheckpointManager {
    checkpoints_dir: std::path::PathBuf,
    max_checkpoints: usize,
}

impl CheckpointManager {
    pub fn new(checkpoints_dir: &Path, max_checkpoints: usize) -> Self {
        Self {
            checkpoints_dir: checkpoints_dir.to_path_buf(),
            max_checkpoints,
        }
    }

    /// Save model checkpoint
    pub fn save_checkpoint(
        &self,
        model_data: &[u8],
        epoch: usize,
        loss: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;

        fs::create_dir_all(&self.checkpoints_dir)?;

        let checkpoint_path = self.checkpoints_dir.join(format!("checkpoint_epoch_{}.bin", epoch));

        // Save model data and metadata
        fs::write(&checkpoint_path, model_data)?;
        log::info!("Saving checkpoint to: {:?}", checkpoint_path);

        // Clean up old checkpoints if we exceed max_checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(())
    }

    /// Load model checkpoint
    pub fn load_checkpoint(
        &self,
        epoch: usize,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let checkpoint_path = self.checkpoints_dir.join(format!("checkpoint_epoch_{}.bin", epoch));

        if !checkpoint_path.exists() {
            return Err(format!("Checkpoint not found: {:?}", checkpoint_path).into());
        }

        let data = std::fs::read(&checkpoint_path)?;
        log::info!("Loading checkpoint from: {:?}", checkpoint_path);

        Ok(data)
    }

    /// Get best checkpoint based on validation metric
    pub fn get_best_checkpoint(&self, metric_name: &str) -> Option<std::path::PathBuf> {
        // Placeholder implementation
        // In practice, you would scan checkpoint files and find the best one
        None
    }

    fn cleanup_old_checkpoints(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Placeholder implementation
        // In practice, you would list checkpoint files and remove old ones
        Ok(())
    }
}

/// Cross-validation utilities
pub struct CrossValidator {
    num_folds: usize,
}

impl CrossValidator {
    pub fn new(num_folds: usize) -> Self {
        Self { num_folds }
    }

    /// Perform k-fold cross-validation
    pub fn cross_validate<F, T>(&self, data: &[T], train_fn: F) -> Vec<f32>
    where
        F: Fn(&[T], &[T]) -> f32,
        T: Clone,
    {
        let fold_size = data.len() / self.num_folds;
        let mut scores = Vec::new();

        for fold in 0..self.num_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == self.num_folds - 1 {
                data.len()
            } else {
                (fold + 1) * fold_size
            };

            let test_data = &data[test_start..test_end];
            let train_data = [&data[..test_start], &data[test_end..]].concat();

            let score = train_fn(&train_data, test_data);
            scores.push(score);
        }

        scores
    }
}