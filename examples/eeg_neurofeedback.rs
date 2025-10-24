//! EEG Neurofeedback Example - Real-time brain state monitoring and feedback

use stream_diffusion_rs::*;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    log::info!("EEG Neurofeedback System - Real-time Brain State Monitoring");

    // Initialize EEG processor
    let mut processor = EEGProcessor::new();

    // Add signal processing filters
    processor.add_filter("notch", DigitalFilter::new(FilterType::BandStop, 4, 50.0, 250.0)); // Remove 50Hz noise
    processor.add_filter("bandpass", DigitalFilter::new(FilterType::BandPass, 4, 1.0, 40.0)); // Keep brain waves

    // Initialize visualizer for real-time feedback
    let visualizer = EEGVisualizer::new(std::path::Path::new("output"));

    // Initialize audiovisual converter
    let converter = EEGToAudiovisualConverter::new(std::path::Path::new("output"));

    // Simulate real-time EEG data stream
    log::info!("Starting neurofeedback session...");

    let session_duration = Duration::from_secs(60); // 1 minute session
    let start_time = Instant::now();
    let mut frame_count = 0;

    while start_time.elapsed() < session_duration {
        // Simulate EEG data acquisition (in practice, this would come from hardware)
        let eeg_data = generate_simulated_eeg_data();

        // Process the EEG data
        let mut processed_data = eeg_data.clone();
        processor.remove_dc_offset(&mut processed_data);

        // Extract frequency bands
        let alpha_power = processor.extract_band_power(&processed_data, FrequencyBand::Alpha)?;
        let beta_power = processor.extract_band_power(&processed_data, FrequencyBand::Beta)?;
        let theta_power = processor.extract_band_power(&processed_data, FrequencyBand::Theta)?;

        // Calculate focus index (beta/alpha ratio)
        let focus_index = calculate_focus_index(&beta_power, &alpha_power);

        // Generate real-time feedback
        if frame_count % 30 == 0 { // Every ~100ms at 250Hz
            log::info!("Focus Index: {:.3}, Alpha: {:.3}, Beta: {:.3}",
                      focus_index,
                      alpha_power.mean().unwrap_or(0.0),
                      beta_power.mean().unwrap_or(0.0));

            // Generate visual feedback based on brain state
            generate_neurofeedback_visualization(&processed_data, focus_index, &visualizer, frame_count)?;

            // Generate audio feedback
            generate_neurofeedback_audio(&processed_data, focus_index)?;
        }

        frame_count += 1;

        // Small delay to simulate real-time processing
        std::thread::sleep(Duration::from_millis(4)); // ~250Hz sampling
    }

    log::info!("Neurofeedback session completed");
    log::info!("Processed {} EEG frames", frame_count);

    Ok(())
}

/// Generate simulated EEG data for demonstration
fn generate_simulated_eeg_data() -> EEGData {
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;

    let num_channels = 32;
    let time_steps = 250; // 1 second at 250Hz
    let epochs = 1;

    // Generate realistic EEG-like signals
    let mut data = ndarray::Array3::<f32>::zeros((num_channels, time_steps, epochs));

    for ch in 0..num_channels {
        // Mix of different frequency components
        for t in 0..time_steps {
            let t_sec = t as f32 / 250.0;

            // Alpha waves (8-12 Hz) - relaxed state
            let alpha = 2.0 * (2.0 * std::f32::consts::PI * 10.0 * t_sec).sin();

            // Beta waves (12-30 Hz) - active thinking
            let beta = 1.5 * (2.0 * std::f32::consts::PI * 20.0 * t_sec).sin();

            // Theta waves (4-8 Hz) - meditative state
            let theta = 1.0 * (2.0 * std::f32::consts::PI * 6.0 * t_sec).sin();

            // Add some noise
            let noise = ndarray::Array1::<f32>::random(1, Normal::new(0.0, 0.5).unwrap())[0];

            data[[ch, t, 0]] = alpha + beta + theta + noise;
        }
    }

    let channel_names = (0..num_channels)
        .map(|i| format!("EEG{:02}", i + 1))
        .collect();

    EEGData::new(data, 250.0, channel_names)
}

/// Calculate focus index from EEG bands
fn calculate_focus_index(beta_power: &ndarray::Array2<f32>, alpha_power: &ndarray::Array2<f32>) -> f32 {
    let beta_mean = beta_power.mean().unwrap_or(1.0);
    let alpha_mean = alpha_power.mean().unwrap_or(1.0);

    if alpha_mean > 0.0 {
        beta_mean / alpha_mean
    } else {
        0.0
    }
}

/// Generate real-time neurofeedback visualization
fn generate_neurofeedback_visualization(
    eeg_data: &EEGData,
    focus_index: f32,
    visualizer: &EEGVisualizer,
    frame_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create focus-based color mapping
    let intensity = (focus_index.min(3.0) / 3.0).max(0.0); // Normalize to 0-1

    // Generate topographic map
    let values: Vec<f32> = (0..eeg_data.num_channels())
        .map(|i| eeg_data.get_channel_epoch(i, 0).mean().unwrap_or(0.0))
        .collect();

    let electrode_names: Vec<&str> = eeg_data.channel_names.iter()
        .map(|s| s.as_str())
        .collect();

    let filename = format!("neurofeedback_topography_{}.png", frame_count);
    visualizer.plot_eeg_topography(&values, &electrode_names, &filename)?;

    // Generate focus meter visualization
    generate_focus_meter(intensity, frame_count, visualizer)?;

    Ok(())
}

/// Generate focus meter visualization
fn generate_focus_meter(intensity: f32, frame_count: usize, visualizer: &EEGVisualizer) -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple focus meter image
    let width = 400;
    let height = 100;
    let mut image_data = vec![0u8; width * height * 3];

    // Draw meter background
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;

            if y > height / 2 - 10 && y < height / 2 + 10 {
                // Meter bar background
                image_data[idx] = 64;     // R
                image_data[idx + 1] = 64; // G
                image_data[idx + 2] = 64; // B
            } else {
                // Background
                image_data[idx] = 32;     // R
                image_data[idx + 1] = 32; // G
                image_data[idx + 2] = 32; // B
            }
        }
    }

    // Draw focus level
    let fill_width = (intensity * (width as f32 - 40.0)) as usize;
    for y in (height / 2 - 8)..=(height / 2 + 8) {
        for x in 20..(20 + fill_width) {
            if x < width && y < height {
                let idx = (y * width + x) * 3;

                // Color based on focus level
                if intensity > 0.7 {
                    image_data[idx] = 0;     // High focus - Green
                    image_data[idx + 1] = 255;
                    image_data[idx + 2] = 0;
                } else if intensity > 0.4 {
                    image_data[idx] = 255;   // Medium focus - Yellow
                    image_data[idx + 1] = 255;
                    image_data[idx + 2] = 0;
                } else {
                    image_data[idx] = 255;   // Low focus - Red
                    image_data[idx + 1] = 0;
                    image_data[idx + 2] = 0;
                }
            }
        }
    }

    // Save focus meter (placeholder - would save to file)
    let filename = format!("focus_meter_{}.png", frame_count);
    log::info!("Generated focus meter: {} (intensity: {:.2})", filename, intensity);

    Ok(())
}

/// Generate audio feedback based on brain state
fn generate_neurofeedback_audio(eeg_data: &EEGData, focus_index: f32) -> Result<(), Box<dyn std::error::Error>> {
    // Generate binaural beats or tones based on focus level
    let base_freq = 200.0; // Base frequency in Hz
    let focus_freq = 10.0 + focus_index * 20.0; // 10-30 Hz based on focus

    // In practice, this would generate actual audio output
    log::debug!("Audio feedback - Base: {:.1}Hz, Focus: {:.1}Hz", base_freq, focus_freq);

    Ok(())
}