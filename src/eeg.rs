//! EEG data analysis and audiovisual conversion utilities

use ndarray::{Array2, Array3, Axis, s};
use plotters::prelude::*;
use std::collections::HashMap;
use std::path::Path;

// Add Python interop
use crate::python::{PythonEnvironment, PythonModel};

/// EEG data structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EEGData {
    pub data: Array3<f32>, // [channels, time_steps, epochs]
    pub sampling_rate: f32,
    pub channel_names: Vec<String>,
    pub metadata: HashMap<String, String>,
}

impl EEGData {
    pub fn new(data: Array3<f32>, sampling_rate: f32, channel_names: Vec<String>) -> Self {
        Self {
            data,
            sampling_rate,
            channel_names,
            metadata: HashMap::new(),
        }
    }

    /// Load EEG data from file (EDF, BDF, etc.)
    pub fn load_from_file(file_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("Loading EEG data from: {:?}", file_path);

        // Placeholder implementation
        // In practice, you would use libraries like edf-rs or mne-rust

        // Dummy data for now
        let data = Array3::<f32>::zeros((32, 1000, 10)); // 32 channels, 1000 time steps, 10 epochs
        let channel_names = (0..32).map(|i| format!("Ch{}", i + 1)).collect();
        let sampling_rate = 250.0;

        Ok(Self::new(data, sampling_rate, channel_names))
    }

    /// Get number of channels
    pub fn num_channels(&self) -> usize {
        self.data.dim().0
    }

    /// Get number of time steps
    pub fn num_time_steps(&self) -> usize {
        self.data.dim().2
    }

    /// Get number of epochs
    pub fn num_epochs(&self) -> usize {
        self.data.dim().2
    }

    /// Get data for specific channel and epoch
    pub fn get_channel_epoch(&self, channel: usize, epoch: usize) -> Array2<f32> {
        self.data.slice(s![channel, .., epoch..epoch+1]).to_owned()
    }

    /// Get time vector
    pub fn get_time_vector(&self) -> Vec<f32> {
        let num_samples = self.num_time_steps();
        (0..num_samples)
            .map(|i| i as f32 / self.sampling_rate)
            .collect()
    }
}

/// EEG signal processing utilities
pub struct EEGProcessor {
    filters: HashMap<String, DigitalFilter>,
    python_model: Option<PythonModel>,
}

impl EEGProcessor {
    pub fn new() -> Self {
        Self {
            filters: HashMap::new(),
            python_model: None,
        }
    }

    /// Initialize Python model for real EEG processing
    pub fn with_python_model(mut self, model_path: &str) -> Self {
        let python_env = PythonEnvironment::default();
        let python_model = PythonModel::new(model_path, "eeg", python_env);
        self.python_model = Some(python_model);
        self
    }

    /// Train an EEG model using Python
    pub fn train_model(&self, training_config: &crate::python::TrainingConfig) -> Result<crate::python::PythonResult, Box<dyn std::error::Error>> {
        if let Some(python_model) = &self.python_model {
            python_model.train_model(training_config)
        } else {
            Err("No Python model configured for training".into())
        }
    }

    /// Fine-tune an EEG model using Python
    pub fn fine_tune_model(&self, fine_tuning_config: &crate::python::FineTuningConfig) -> Result<crate::python::PythonResult, Box<dyn std::error::Error>> {
        if let Some(python_model) = &self.python_model {
            python_model.fine_tune_model(fine_tuning_config)
        } else {
            Err("No Python model configured for fine-tuning".into())
        }
    }

    /// Evaluate an EEG model using Python
    pub fn evaluate_model(&self, evaluation_config: &crate::python::EvaluationConfig) -> Result<crate::python::PythonResult, Box<dyn std::error::Error>> {
        if let Some(python_model) = &self.python_model {
            python_model.evaluate_model(evaluation_config)
        } else {
            Err("No Python model configured for evaluation".into())
        }
    }

    /// Add a digital filter
    pub fn add_filter(&mut self, name: &str, filter: DigitalFilter) {
        self.filters.insert(name.to_string(), filter);
    }

    /// Apply filter to EEG data
    pub fn apply_filter(&self, data: &mut EEGData, filter_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(filter) = self.filters.get(filter_name) {
            for channel in 0..data.num_channels() {
                for epoch in 0..data.num_epochs() {
                    let mut channel_data = data.get_channel_epoch(channel, epoch);
                    filter.apply(&mut channel_data);
                    // Put filtered data back
                    for (i, &val) in channel_data.iter().enumerate() {
                        data.data[[channel, i, epoch]] = val;
                    }
                }
            }
            Ok(())
        } else {
            Err(format!("Filter '{}' not found", filter_name).into())
        }
    }

    /// Compute statistical features for EEG data
    pub fn compute_statistics(&self, data: &EEGData) -> EEGStatistics {
        let mut stats = EEGStatistics {
            channel_means: Vec::new(),
            channel_stds: Vec::new(),
            channel_skewness: Vec::new(),
            channel_kurtosis: Vec::new(),
            global_mean: 0.0,
            global_std: 0.0,
        };

        let mut all_values = Vec::new();

        for channel in 0..data.num_channels() {
            let mut channel_values = Vec::new();

            for epoch in 0..data.num_epochs() {
                let channel_data = data.get_channel_epoch(channel, epoch);
                channel_values.extend(channel_data.iter().cloned());
            }

            let mean = channel_values.iter().sum::<f32>() / channel_values.len() as f32;
            let variance = channel_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / channel_values.len() as f32;
            let std = variance.sqrt();

            stats.channel_means.push(mean);
            stats.channel_stds.push(std);

            // Simplified skewness and kurtosis
            stats.channel_skewness.push(0.0);
            stats.channel_kurtosis.push(0.0);

            all_values.extend(channel_values);
        }

        stats.global_mean = all_values.iter().sum::<f32>() / all_values.len() as f32;
        stats.global_std = (all_values.iter().map(|x| (x - stats.global_mean).powi(2)).sum::<f32>() / all_values.len() as f32).sqrt();

        stats
    }

    /// Detect artifacts in EEG data
    pub fn detect_artifacts(&self, data: &EEGData, threshold: f32) -> Vec<ArtifactDetection> {
        let mut artifacts = Vec::new();

        for channel in 0..data.num_channels() {
            for epoch in 0..data.num_epochs() {
                let channel_data = data.get_channel_epoch(channel, epoch);
                let amplitude = channel_data.fold(0.0f32, |acc, &x| acc.max(x.abs()));

                if amplitude > threshold {
                    artifacts.push(ArtifactDetection {
                        channel,
                        epoch,
                        amplitude,
                        artifact_type: ArtifactType::HighAmplitude,
                    });
                }
            }
        }

        artifacts
    }

    /// Perform independent component analysis (ICA) - placeholder
    pub fn perform_ica(&self, data: &EEGData, num_components: usize) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
        // Placeholder for ICA implementation
        // In practice, this would use a proper ICA algorithm like FastICA
        log::info!("Performing ICA with {} components (placeholder)", num_components);

        // Return dummy components
        Ok(Array3::<f32>::zeros((num_components, data.num_time_steps(), data.num_epochs())))
    }

    /// Remove DC offset (baseline correction)
    pub fn remove_dc_offset(&self, data: &mut EEGData) {
        for channel in 0..data.num_channels() {
            for epoch in 0..data.num_epochs() {
                let channel_data = data.get_channel_epoch(channel, epoch);
                let mean = channel_data.mean().unwrap_or(0.0);
                let corrected = &channel_data - mean;

                for (i, &val) in corrected.iter().enumerate() {
                    data.data[[channel, i, epoch]] = val;
                }
            }
        }
    }

    /// Compute power spectral density
    pub fn compute_psd(&self, data: &EEGData, channel: usize, epoch: usize) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let channel_data = data.get_channel_epoch(channel, epoch);
        let signal: Vec<f32> = channel_data.iter().cloned().collect();

        // Simple FFT-based PSD computation (placeholder)
        // In practice, you would use a proper FFT library
        let freqs: Vec<f32> = (0..signal.len()/2)
            .map(|i| i as f32 * data.sampling_rate / signal.len() as f32)
            .collect();

        let psd: Vec<f32> = signal.iter().map(|&x| x * x).collect();

        Ok((freqs, psd))
    }

    /// Extract frequency band power with Python processing if available
    pub fn extract_band_power(&self, data: &EEGData, band: FrequencyBand) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        // Use Python model if available for real processing
        if let Some(python_model) = &self.python_model {
            // Save EEG data to temporary file for Python processing
            let temp_path = std::env::temp_dir().join("eeg_data.npy");
            // In a real implementation, you would save the actual data to this file
            
            let result = python_model.process_eeg(temp_path.to_str().unwrap())?;
            if result.success {
                log::info!("Processed EEG data with Python model");
                // In a real implementation, you would parse the result and return actual data
            }
        }
        
        // Fallback to existing implementation
        let (low_freq, high_freq) = band.range();
        let mut band_powers = Array2::<f32>::zeros((data.num_channels(), data.num_epochs()));

        for channel in 0..data.num_channels() {
            for epoch in 0..data.num_epochs() {
                let (freqs, psd) = self.compute_psd(data, channel, epoch)?;

                let band_power: f32 = freqs.iter().zip(psd.iter())
                    .filter(|(&f, _)| f >= low_freq && f <= high_freq)
                    .map(|(_, &p)| p)
                    .sum();

                band_powers[[channel, epoch]] = band_power;
            }
        }

        Ok(band_powers)
    }

    /// Calculate connectivity with Python processing if available
    pub fn calculate_connectivity(&self, data: &EEGData) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Use Python model if available for real processing
        if let Some(python_model) = &self.python_model {
            // Save EEG data to temporary file for Python processing
            let temp_path = std::env::temp_dir().join("eeg_data.npy");
            // In a real implementation, you would save the actual data to this file
            
            let result = python_model.process_eeg(temp_path.to_str().unwrap())?;
            if result.success {
                log::info!("Calculated connectivity with Python model");
                // In a real implementation, you would parse the result and return actual data
                if let Some(ref json_data) = result.data {
                    if let Some(connectivity) = json_data.get("connectivity") {
                        if let Some(arr) = connectivity.as_array() {
                            return Ok(arr.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect());
                        }
                    }
                }
            }
        }
        
        // Fallback to simple correlation-based connectivity
        let mut connectivity = Vec::new();
        let channel_count = data.num_channels();
        
        // Create a simple correlation matrix (placeholder)
        for i in 0..channel_count {
            for j in 0..channel_count {
                if i == j {
                    connectivity.push(1.0); // Self-correlation is 1.0
                } else {
                    connectivity.push(0.5); // Placeholder correlation value
                }
            }
        }
        
        Ok(connectivity)
    }

    /// Calculate complexity with Python processing if available
    pub fn calculate_complexity(&self, data: &EEGData) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Use Python model if available for real processing
        if let Some(python_model) = &self.python_model {
            // Save EEG data to temporary file for Python processing
            let temp_path = std::env::temp_dir().join("eeg_data.npy");
            // In a real implementation, you would save the actual data to this file
            
            let result = python_model.process_eeg(temp_path.to_str().unwrap())?;
            if result.success {
                log::info!("Calculated complexity with Python model");
                // In a real implementation, you would parse the result and return actual data
                if let Some(ref json_data) = result.data {
                    if let Some(complexity) = json_data.get("complexity") {
                        if let Some(arr) = complexity.as_array() {
                            return Ok(arr.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect());
                        }
                    }
                }
            }
        }
        
        // Fallback to simple standard deviation-based complexity
        let mut complexity = Vec::new();
        
        for channel in 0..data.num_channels() {
            let mut channel_values = Vec::new();
            for epoch in 0..data.num_epochs() {
                let channel_data = data.get_channel_epoch(channel, epoch);
                channel_values.extend(channel_data.iter().cloned());
            }
            
            // Calculate standard deviation as complexity measure
            let mean = channel_values.iter().sum::<f32>() / channel_values.len() as f32;
            let variance = channel_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / channel_values.len() as f32;
            let std = variance.sqrt();
            
            complexity.push(std);
        }
        
        Ok(complexity)
    }
}

/// Digital filter for signal processing
#[derive(Debug, Clone)]
pub struct DigitalFilter {
    pub filter_type: FilterType,
    pub order: usize,
    pub cutoff_freq: f32,
    pub sampling_rate: f32,
}

impl DigitalFilter {
    pub fn new(filter_type: FilterType, order: usize, cutoff_freq: f32, sampling_rate: f32) -> Self {
        Self {
            filter_type,
            order,
            cutoff_freq,
            sampling_rate,
        }
    }

    /// Apply filter to signal (placeholder implementation)
    pub fn apply(&self, signal: &mut Array2<f32>) {
        // Simple moving average as placeholder
        // In practice, you would implement proper IIR/FIR filters
        let window_size = 5;
        let mut filtered = signal.clone();

        for i in window_size..signal.nrows() {
            let sum: f32 = (0..window_size).map(|j| signal[[0, i - j]]).sum();
            filtered[[0, i]] = sum / window_size as f32;
        }

        *signal = filtered;
    }
}

/// Filter types
#[derive(Debug, Clone)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    BandStop,
}

/// EEG frequency bands
#[derive(Debug, Clone)]
pub enum FrequencyBand {
    Delta,    // 0.5-4 Hz
    Theta,    // 4-8 Hz
    Alpha,    // 8-12 Hz
    Beta,     // 12-30 Hz
    Gamma,    // 30-100 Hz
}

/// EEG statistics
#[derive(Debug)]
pub struct EEGStatistics {
    pub channel_means: Vec<f32>,
    pub channel_stds: Vec<f32>,
    pub channel_skewness: Vec<f32>,
    pub channel_kurtosis: Vec<f32>,
    pub global_mean: f32,
    pub global_std: f32,
}

/// Artifact detection result
#[derive(Debug)]
pub struct ArtifactDetection {
    pub channel: usize,
    pub epoch: usize,
    pub amplitude: f32,
    pub artifact_type: ArtifactType,
}

/// Types of artifacts
#[derive(Debug)]
pub enum ArtifactType {
    HighAmplitude,
    MuscleArtifact,
    EyeBlink,
    ElectrodeNoise,
}

impl FrequencyBand {
    pub fn range(&self) -> (f32, f32) {
        match self {
            Self::Delta => (0.5, 4.0),
            Self::Theta => (4.0, 8.0),
            Self::Alpha => (8.0, 12.0),
            Self::Beta => (12.0, 30.0),
            Self::Gamma => (30.0, 100.0),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Delta => "Delta",
            Self::Theta => "Theta",
            Self::Alpha => "Alpha",
            Self::Beta => "Beta",
            Self::Gamma => "Gamma",
        }
    }
}

/// EEG visualization utilities
pub struct EEGVisualizer {
    output_dir: std::path::PathBuf,
}

impl EEGVisualizer {
    pub fn new(output_dir: &Path) -> Self {
        Self {
            output_dir: output_dir.to_path_buf(),
        }
    }

    /// Plot EEG signal for a channel
    pub fn plot_channel(&self, data: &EEGData, channel: usize, epoch: usize, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let channel_data = data.get_channel_epoch(channel, epoch);
        let time_vector = data.get_time_vector();

        let output_path = self.output_dir.join(filename);

        let mut buffer = vec![0u8; 800 * 600 * 3];
        let root = BitMapBackend::with_buffer(&mut buffer, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("EEG Channel {} - Epoch {}", data.channel_names[channel], epoch), ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0.0..*time_vector.last().unwrap_or(&1.0), channel_data.fold(f32::INFINITY, |a, &b| a.min(b))..channel_data.fold(f32::NEG_INFINITY, |a, &b| a.max(b)))?;

        chart.configure_mesh().draw()?;

        chart.draw_series(plotters::series::LineSeries::new(
            time_vector.iter().zip(channel_data.iter()).map(|(&t, &v)| (t, v)),
            &plotters::style::RED,
        ))?;

        root.present()?;
        Ok(())
    }

    /// Plot power spectral density
    pub fn plot_psd(&self, freqs: &[f32], psd: &[f32], channel_name: &str, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let output_path = self.output_dir.join(filename);

        let mut buffer = vec![0u8; 800 * 600 * 3];
        let root = BitMapBackend::with_buffer(&mut buffer, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("PSD - {}", channel_name), ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0.0..*freqs.last().unwrap_or(&100.0), 0.0..psd.iter().fold(0.0f32, |a, &b| a.max(b)))?;

        chart.configure_mesh().draw()?;

        chart.draw_series(plotters::series::LineSeries::new(
            freqs.iter().zip(psd.iter()).map(|(&f, &p)| (f, p)),
            &plotters::style::BLUE,
        ))?;

        root.present()?;
        Ok(())
    }

    /// Plot topographic map of EEG data
    pub fn plot_topography(&self, data: &EEGData, epoch: usize, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Simplified topography plot (placeholder)
        // In practice, you would need electrode positions and interpolation

        let output_path = self.output_dir.join(filename);

        let mut buffer = vec![0u8; 600 * 600 * 3];
        let root = BitMapBackend::with_buffer(&mut buffer, (600, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("EEG Topography", ("sans-serif", 20))
            .build_cartesian_2d(-1.0..1.0, -1.0..1.0)?;

        chart.configure_mesh().draw()?;

        // Plot electrode positions (simplified 10-20 system)
        let electrode_positions = vec![
            (0.0, 0.8),   // Fz
            (-0.6, 0.6),  // F3
            (0.6, 0.6),   // F4
            (-0.8, 0.0),  // C3
            (0.8, 0.0),   // C4
            (-0.6, -0.6), // P3
            (0.6, -0.6),  // P4
            (0.0, -0.8),  // Pz
        ];

        for (i, &(x, y)) in electrode_positions.iter().enumerate() {
            if i < data.num_channels() {
                let value = data.data[[i, 0, epoch]]; // First time point
                let color = if value > 0.0 { &plotters::style::RED } else { &plotters::style::BLUE };
                chart.draw_series(std::iter::once(plotters::element::Circle::new((x, y), 5, color)))?;

                chart.draw_series(std::iter::once(plotters::element::Text::new(
                    format!("Ch{}", i + 1),
                    (x, y + 0.05),
                    plotters::style::FontDesc::new(plotters::style::FontFamily::SansSerif, 10.0, plotters::style::FontStyle::Normal).color(&plotters::style::BLACK),
                )))?;
            }
        }

        root.present()?;
        Ok(())
    }
}

/// EEG to audiovisual conversion
pub struct EEGToAudiovisualConverter {
    processor: EEGProcessor,
    visualizer: EEGVisualizer,
    audio_generator: AudioGenerator,
}

impl EEGToAudiovisualConverter {
    pub fn new(output_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            processor: EEGProcessor::new(),
            visualizer: EEGVisualizer::new(output_dir),
            audio_generator: AudioGenerator::new(),
        })
    }

    /// Convert EEG data to audiovisual representation
    pub fn convert(&self, eeg_data: &EEGData) -> Result<AudiovisualData, Box<dyn std::error::Error>> {
        // Extract features from EEG
        let features = self.extract_features(eeg_data)?;

        // Generate visual representation
        let visual_data = self.generate_visual(eeg_data, &features)?;

        // Generate audio representation
        let audio_data = self.generate_audio(&features)?;

        Ok(AudiovisualData {
            visual: visual_data,
            audio: audio_data,
            features,
        })
    }

    fn extract_features(&self, eeg_data: &EEGData) -> Result<EEGFeatures, Box<dyn std::error::Error>> {
        let mut features = EEGFeatures {
            band_powers: HashMap::new(),
            connectivity: vec![0.0; eeg_data.num_channels() * eeg_data.num_channels()],
            complexity: Vec::new(),
        };

        // Extract frequency band powers
        for band in &[FrequencyBand::Alpha, FrequencyBand::Beta, FrequencyBand::Theta] {
            let powers = self.processor.extract_band_power(eeg_data, band.clone())?;
            let powers_vec: Vec<f32> = powers.iter().cloned().collect();
            features.band_powers.insert(band.name().to_string(), powers_vec);
        }

        // Compute connectivity matrix (simplified correlation)
        let mut connectivity_vec = Vec::new();
        for i in 0..eeg_data.num_channels() {
            for j in 0..eeg_data.num_channels() {
                let channel_i = eeg_data.data.slice(s![i, .., 0]);
                let channel_j = eeg_data.data.slice(s![j, .., 0]);
                // Simple correlation coefficient
                let corr = self.compute_correlation(&channel_i, &channel_j);
                connectivity_vec.push(corr);
            }
        }
        features.connectivity = connectivity_vec;

        // Compute complexity measures (Hjorth parameters)
        for channel in 0..eeg_data.num_channels() {
            let channel_data = eeg_data.get_channel_epoch(channel, 0);
            let complexity = self.compute_hjorth_complexity(&channel_data);
            features.complexity.push(complexity);
        }

        Ok(features)
    }

    fn generate_visual(&self, eeg_data: &EEGData, features: &EEGFeatures) -> Result<VisualData, Box<dyn std::error::Error>> {
        // Generate visual representation based on EEG features
        // This could be fractal patterns, color mappings, etc.

        let width = 512;
        let height = 512;
        let mut image_data = vec![0u8; width * height * 3];

        // Map EEG features to visual elements
        for (band_name, powers) in &features.band_powers {
            // Use band powers to modulate colors/intensity
            let intensity = powers.iter().sum::<f32>() / powers.len() as f32 * 255.0;
            let color_offset = match band_name.as_str() {
                "Alpha" => 0,
                "Beta" => 1,
                "Theta" => 2,
                _ => 0,
            };

            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 3;
                    image_data[idx + color_offset] = (intensity as u8).min(255);
                }
            }
        }

        Ok(VisualData {
            image_data,
            width,
            height,
        })
    }

    fn generate_audio(&self, features: &EEGFeatures) -> Result<AudioData, Box<dyn std::error::Error>> {
        self.audio_generator.generate_from_features(features)
    }


    /// Create real-time EEG processor for streaming data
    pub fn create_realtime_processor(&self, buffer_size: usize) -> RealtimeEEGProcessor {
        RealtimeEEGProcessor::new(buffer_size)
    }

    fn compute_correlation(&self, a: &ndarray::ArrayView1<f32>, b: &ndarray::ArrayView1<f32>) -> f32 {
        let mean_a = a.mean().unwrap_or(0.0);
        let mean_b = b.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut sum_a_sq = 0.0;
        let mut sum_b_sq = 0.0;

        for (&x, &y) in a.iter().zip(b.iter()) {
            let dx = x - mean_a;
            let dy = y - mean_b;
            numerator += dx * dy;
            sum_a_sq += dx * dx;
            sum_b_sq += dy * dy;
        }

        if sum_a_sq == 0.0 || sum_b_sq == 0.0 {
            0.0
        } else {
            numerator / (sum_a_sq * sum_b_sq).sqrt()
        }
    }

    fn compute_hjorth_complexity(&self, signal: &Array2<f32>) -> f32 {
        // Simplified Hjorth complexity (mobility * mobility of derivative)
        let derivative: Vec<f32> = signal.iter().zip(signal.iter().skip(1))
            .map(|(&a, &b)| b - a)
            .collect();

        let signal_std = signal.std(0.0);
        let deriv_std = derivative.iter().cloned().collect::<Vec<f32>>().iter().map(|&x| (x - 0.0).powi(2)).sum::<f32>().sqrt() / derivative.len() as f32;

        if signal_std == 0.0 {
            0.0
        } else {
            (deriv_std / signal_std).powi(2)
        }
    }
}

/// Extracted EEG features
#[derive(Debug, serde::Serialize)]
pub struct EEGFeatures {
    pub band_powers: HashMap<String, Vec<f32>>,
    pub connectivity: Vec<f32>,
    pub complexity: Vec<f32>,
}

/// Generated visual data
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct VisualData {
    pub image_data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// Generated audio data
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Combined audiovisual data
#[derive(Debug)]
pub struct AudiovisualData {
    pub visual: VisualData,
    pub audio: AudioData,
    pub features: EEGFeatures,
}

/// Audio generation utilities
pub struct AudioGenerator {
    sample_rate: u32,
}

impl AudioGenerator {
    pub fn new() -> Self {
        Self { sample_rate: 44100 }
    }

    pub fn generate_from_features(&self, features: &EEGFeatures) -> Result<AudioData, Box<dyn std::error::Error>> {
        let duration = 2.0; // 2 seconds
        let num_samples = (duration * self.sample_rate as f32) as usize;
        let mut samples = vec![0.0f32; num_samples];

        // Generate audio based on EEG features
        for (band_name, powers) in &features.band_powers {
            let frequency = match band_name.as_str() {
                "Alpha" => 10.0, // Alpha rhythm
                "Beta" => 20.0,  // Beta rhythm
                "Theta" => 6.0,   // Theta rhythm
                _ => 10.0,
            };

            let amplitude = powers.iter().sum::<f32>() / powers.len() as f32 * 0.1;

            for i in 0..num_samples {
                let t = i as f32 / self.sample_rate as f32;
                samples[i] += amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin();
            }
        }

        Ok(AudioData {
            samples,
            sample_rate: self.sample_rate,
        })
    }

    /// Generate binaural beats for brainwave entrainment
    pub fn generate_binaural_beats(&self, base_freq: f32, beat_freq: f32, duration: f32) -> AudioData {
        let num_samples = (duration * self.sample_rate as f32) as usize;
        let mut left_channel = vec![0.0f32; num_samples];
        let mut right_channel = vec![0.0f32; num_samples];

        for i in 0..num_samples {
            let t = i as f32 / self.sample_rate as f32;
            left_channel[i] = (2.0 * std::f32::consts::PI * base_freq * t).sin();
            right_channel[i] = (2.0 * std::f32::consts::PI * (base_freq + beat_freq) * t).sin();
        }

        // Interleave channels for stereo
        let mut samples = Vec::new();
        for i in 0..num_samples {
            samples.push(left_channel[i]);
            samples.push(right_channel[i]);
        }

        AudioData {
            samples,
            sample_rate: self.sample_rate,
        }
    }
}

/// Real-time EEG processor for streaming data
pub struct RealtimeEEGProcessor {
    buffer: Vec<f32>,
    buffer_size: usize,
    sampling_rate: f32,
    filters: HashMap<String, DigitalFilter>,
}

impl RealtimeEEGProcessor {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer: Vec::new(),
            buffer_size,
            sampling_rate: 250.0,
            filters: HashMap::new(),
        }
    }

    /// Add new EEG sample
    pub fn add_sample(&mut self, sample: f32) {
        self.buffer.push(sample);
        if self.buffer.len() > self.buffer_size {
            self.buffer.remove(0);
        }
    }

    /// Process buffer and extract features
    pub fn process_buffer(&self) -> Result<RealtimeEEGFeatures, Box<dyn std::error::Error>> {
        if self.buffer.len() < 100 { // Minimum buffer size
            return Err("Insufficient buffer size".into());
        }

        let signal = Array2::from_shape_vec((1, self.buffer.len()), self.buffer.clone()).unwrap();

        // Compute basic features
        let mean = signal.mean().unwrap_or(0.0);
        let std = signal.std(0.0);

        // Simple band power estimation (placeholder)
        let alpha_power = self.estimate_band_power(&signal, 8.0, 12.0)?;
        let beta_power = self.estimate_band_power(&signal, 12.0, 30.0)?;

        Ok(RealtimeEEGFeatures {
            mean,
            std,
            alpha_power,
            beta_power,
            buffer_size: self.buffer.len(),
        })
    }

    fn estimate_band_power(&self, signal: &Array2<f32>, low_freq: f32, high_freq: f32) -> Result<f32, Box<dyn std::error::Error>> {
        // Simplified band power estimation
        let signal_vec: Vec<f32> = signal.iter().cloned().collect();
        let power: f32 = signal_vec.iter().map(|&x| x * x).sum();
        Ok(power / signal_vec.len() as f32)
    }

    /// Get current buffer as slice
    pub fn get_buffer(&self) -> &[f32] {
        &self.buffer
    }

    /// Clear buffer
    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
    }
}

/// Real-time EEG features
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct RealtimeEEGFeatures {
    pub mean: f32,
    pub std: f32,
    pub alpha_power: f32,
    pub beta_power: f32,
    pub buffer_size: usize,
}