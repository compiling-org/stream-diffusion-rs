//! Example: EEG data analysis and audiovisual conversion

use stream_diffusion_rs::*;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    log::info!("Starting EEG analysis example");

    // Load EEG data (placeholder - in practice, load from EDF/BDF file)
    let eeg_data = EEGData::new(
        ndarray::Array3::<f32>::zeros((32, 1000, 10)), // 32 channels, 1000 time points, 10 epochs
        250.0, // 250 Hz sampling rate
        (0..32).map(|i| format!("EEG{:02}", i + 1)).collect(),
    );

    log::info!("Loaded EEG data: {} channels, {} time points, {} epochs",
               eeg_data.num_channels(), eeg_data.num_time_steps(), eeg_data.num_epochs());

    // Initialize EEG processor
    let mut processor = EEGProcessor::new();

    // Add filters
    processor.add_filter("notch", DigitalFilter::new(FilterType::BandStop, 4, 50.0, 250.0)); // 50Hz notch
    processor.add_filter("bandpass", DigitalFilter::new(FilterType::BandPass, 4, 1.0, 40.0)); // 1-40Hz bandpass

    // Apply preprocessing
    processor.remove_dc_offset(&mut eeg_data.clone());
    processor.apply_filter(&mut eeg_data.clone(), "notch")?;
    processor.apply_filter(&mut eeg_data.clone(), "bandpass")?;

    // Extract frequency band powers
    let alpha_power = processor.extract_band_power(&eeg_data, FrequencyBand::Alpha)?;
    let beta_power = processor.extract_band_power(&eeg_data, FrequencyBand::Beta)?;
    let theta_power = processor.extract_band_power(&eeg_data, FrequencyBand::Theta)?;

    log::info!("Extracted frequency band powers:");
    log::info!("Alpha power shape: {:?}", alpha_power.dim());
    log::info!("Beta power shape: {:?}", beta_power.dim());
    log::info!("Theta power shape: {:?}", theta_power.dim());

    // Initialize visualizer
    let output_dir = Path::new("output/eeg_analysis");
    std::fs::create_dir_all(output_dir)?;
    let visualizer = EEGVisualizer::new(output_dir);

    // Plot EEG signals
    for channel in 0..3 { // Plot first 3 channels
        for epoch in 0..2 { // Plot first 2 epochs
            visualizer.plot_channel(&eeg_data, channel, epoch,
                &format!("eeg_channel_{}_epoch_{}.png", channel, epoch))?;
        }
    }

    // Plot power spectral density for first channel
    let (freqs, psd) = processor.compute_psd(&eeg_data, 0, 0)?;
    visualizer.plot_psd(&freqs, &psd, "EEG001", "psd_channel_0.png")?;

    // Plot topographic map
    visualizer.plot_topography(&eeg_data, 0, "topography_epoch_0.png")?;

    // Initialize audiovisual converter
    let converter = EEGToAudiovisualConverter::new(output_dir);

    // Extract features and convert to audiovisual
    let audiovisual_data = converter.convert(&eeg_data)?;

    log::info!("Generated audiovisual data:");
    log::info!("Visual: {}x{} image", audiovisual_data.visual.width, audiovisual_data.visual.height);
    log::info!("Audio: {} samples at {} Hz", audiovisual_data.audio.samples.len(), audiovisual_data.audio.sample_rate);

    // Save audio data (placeholder - in practice, save as WAV)
    log::info!("Audio features extracted: {} bands", audiovisual_data.features.band_powers.len());
    log::info!("Connectivity matrix shape: {:?}", audiovisual_data.features.connectivity.dim());

    // Plot connectivity matrix
    let plotter = Plotter::new(output_dir);
    let mut matrix_data = vec![];
    for i in 0..audiovisual_data.features.connectivity.nrows() {
        for j in 0..audiovisual_data.features.connectivity.ncols() {
            matrix_data.push(audiovisual_data.features.connectivity[[i, j]]);
        }
    }

    // Create a simple connectivity visualization
    let connectivity_array = ndarray::Array2::from_shape_vec(
        (audiovisual_data.features.connectivity.nrows(), audiovisual_data.features.connectivity.ncols()),
        matrix_data
    )?;

    // Save connectivity data as JSON for visualization
    let connectivity_json = serde_json::json!({
        "connectivity": connectivity_array,
        "channel_names": eeg_data.channel_names,
    });

    std::fs::write(output_dir.join("connectivity.json"),
                   serde_json::to_string_pretty(&connectivity_json)?)?;

    log::info!("EEG analysis complete. Results saved to: {:?}", output_dir);

    Ok(())
}