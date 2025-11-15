//! Real-time audio synthesis for multimodal AI applications
//!
//! This module provides implementations for real-time audio synthesis,
//! including oscillators, filters, effects, and audio processing for
//! integration with EEG data and other sensor inputs.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Audio sample format
pub type Sample = f32;

/// Audio buffer for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBuffer {
    pub samples: Vec<Sample>,
    pub sample_rate: f32,
    pub channels: usize,
}

impl AudioBuffer {
    /// Create a new audio buffer
    pub fn new(sample_rate: f32, channels: usize, duration_seconds: f32) -> Self {
        let num_samples = (sample_rate * duration_seconds) as usize;
        Self {
            samples: vec![0.0; num_samples * channels],
            sample_rate,
            channels,
        }
    }

    /// Get the number of samples per channel
    pub fn samples_per_channel(&self) -> usize {
        self.samples.len() / self.channels
    }

    /// Get a mutable slice for a specific channel
    pub fn channel_mut(&mut self, channel: usize) -> &mut [Sample] {
        let samples_per_channel = self.samples_per_channel();
        &mut self.samples[channel * samples_per_channel..(channel + 1) * samples_per_channel]
    }

    /// Get a slice for a specific channel
    pub fn channel(&self, channel: usize) -> &[Sample] {
        let samples_per_channel = self.samples_per_channel();
        &self.samples[channel * samples_per_channel..(channel + 1) * samples_per_channel]
    }
}

/// Waveform types for oscillators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
    Noise,
}

/// Audio oscillator for sound generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Oscillator {
    pub waveform: Waveform,
    pub frequency: f32,
    pub amplitude: f32,
    pub phase: f32,
    pub detune: f32,
}

impl Oscillator {
    /// Create a new oscillator
    pub fn new(waveform: Waveform, frequency: f32, amplitude: f32) -> Self {
        Self {
            waveform,
            frequency,
            amplitude,
            phase: 0.0,
            detune: 0.0,
        }
    }

    /// Generate audio samples
    pub fn generate(&mut self, buffer: &mut [Sample], sample_rate: f32) {
        let phase_increment = (2.0 * PI * self.frequency) / sample_rate;
        
        for sample in buffer.iter_mut() {
            *sample += self.generate_sample() * self.amplitude;
            self.phase += phase_increment;
            
            // Keep phase in [0, 2*PI] range
            while self.phase >= 2.0 * PI {
                self.phase -= 2.0 * PI;
            }
        }
    }

    /// Generate a single sample
    fn generate_sample(&self) -> Sample {
        match self.waveform {
            Waveform::Sine => self.phase.sin(),
            Waveform::Square => {
                if self.phase < PI {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Sawtooth => {
                (2.0 * self.phase / (2.0 * PI)) - 1.0
            }
            Waveform::Triangle => {
                let norm_phase = self.phase / (2.0 * PI);
                if norm_phase < 0.25 {
                    4.0 * norm_phase
                } else if norm_phase < 0.75 {
                    2.0 - 4.0 * norm_phase
                } else {
                    4.0 * norm_phase - 4.0
                }
            }
            Waveform::Noise => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                rng.gen::<f32>() * 2.0 - 1.0
            }
        }
    }

    /// Set frequency with optional detune
    pub fn set_frequency(&mut self, frequency: f32, detune_cents: f32) {
        self.frequency = frequency;
        self.detune = detune_cents;
        // Apply detune if needed
        if detune_cents != 0.0 {
            let detune_factor = 2.0f32.powf(detune_cents / 1200.0);
            self.frequency *= detune_factor;
        }
    }
}

/// Filter types for audio processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
}

/// Simple filter implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Filter {
    filter_type: FilterType,
    cutoff: f32,
    resonance: f32,
    sample_rate: f32,
    // Filter state
    x1: Sample,
    x2: Sample,
    y1: Sample,
    y2: Sample,
}

impl Filter {
    /// Create a new filter
    pub fn new(filter_type: FilterType, cutoff: f32, resonance: f32, sample_rate: f32) -> Self {
        Self {
            filter_type,
            cutoff,
            resonance,
            sample_rate,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Process audio samples through the filter
    pub fn process(&mut self, buffer: &mut [Sample]) {
        // Calculate filter coefficients
        let c = (PI * self.cutoff / self.sample_rate).tan();
        let d = 2.0 * (1.0 - self.resonance).sqrt();
        let q = match self.filter_type {
            FilterType::LowPass => 1.0 / (1.0 + d * c + c * c),
            FilterType::HighPass => 1.0 / (1.0 + d * c + c * c),
            FilterType::BandPass => 1.0 / (1.0 + d * c + c * c),
            FilterType::Notch => 1.0 / (1.0 + c * c),
        };

        let (a0, a1, a2, b1, b2) = match self.filter_type {
            FilterType::LowPass => (
                q * c * c,
                2.0 * q * c * c,
                q * c * c,
                2.0 * q * (c * c - 1.0),
                q * (1.0 - d * c + c * c),
            ),
            FilterType::HighPass => (
                q,
                -2.0 * q,
                q,
                2.0 * q * (c * c - 1.0),
                q * (1.0 - d * c + c * c),
            ),
            FilterType::BandPass => (
                q * c,
                0.0,
                -q * c,
                2.0 * q * (c * c - 1.0),
                q * (1.0 - d * c + c * c),
            ),
            FilterType::Notch => (
                q * (1.0 + c * c),
                -2.0 * q * (1.0 - c * c),
                q * (1.0 + c * c),
                2.0 * q * (1.0 - c * c),
                q * (1.0 - c * c + c * c),
            ),
        };

        // Apply filter to each sample
        for sample in buffer.iter_mut() {
            let x0 = *sample;
            let y0 = a0 * x0 + a1 * self.x1 + a2 * self.x2 - b1 * self.y1 - b2 * self.y2;

            // Update state
            self.x2 = self.x1;
            self.x1 = x0;
            self.y2 = self.y1;
            self.y1 = y0;

            *sample = y0;
        }
    }
}

/// Envelope generator for amplitude modulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    sample_rate: f32,
    state: EnvelopeState,
    current_time: f32,
    current_level: f32,
}

/// Envelope states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnvelopeState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl Envelope {
    /// Create a new envelope generator
    pub fn new(attack: f32, decay: f32, sustain: f32, release: f32, sample_rate: f32) -> Self {
        Self {
            attack,
            decay,
            sustain,
            release,
            sample_rate,
            state: EnvelopeState::Idle,
            current_time: 0.0,
            current_level: 0.0,
        }
    }

    /// Trigger the envelope
    pub fn trigger(&mut self) {
        self.state = EnvelopeState::Attack;
        self.current_time = 0.0;
        self.current_level = 0.0;
    }

    /// Release the envelope
    pub fn release(&mut self) {
        self.state = EnvelopeState::Release;
        self.current_time = 0.0;
    }

    /// Generate envelope values
    pub fn generate(&mut self, buffer: &mut [Sample]) {
        for sample in buffer.iter_mut() {
            let dt = 1.0 / self.sample_rate;
            self.current_time += dt;

            match self.state {
                EnvelopeState::Idle => {
                    self.current_level = 0.0;
                }
                EnvelopeState::Attack => {
                    self.current_level = self.current_time / self.attack;
                    if self.current_time >= self.attack {
                        self.state = EnvelopeState::Decay;
                        self.current_time = 0.0;
                    }
                }
                EnvelopeState::Decay => {
                    self.current_level = 1.0 - (1.0 - self.sustain) * (self.current_time / self.decay);
                    if self.current_time >= self.decay {
                        self.state = EnvelopeState::Sustain;
                        self.current_time = 0.0;
                    }
                }
                EnvelopeState::Sustain => {
                    self.current_level = self.sustain;
                }
                EnvelopeState::Release => {
                    self.current_level = self.sustain * (1.0 - self.current_time / self.release);
                    if self.current_time >= self.release {
                        self.state = EnvelopeState::Idle;
                        self.current_time = 0.0;
                        self.current_level = 0.0;
                    }
                }
            }

            *sample *= self.current_level;
        }
    }
}

/// Audio synthesizer combining oscillators, filters, and envelopes
pub struct Synthesizer {
    oscillators: Vec<Oscillator>,
    filters: Vec<Filter>,
    envelope: Envelope,
    sample_rate: f32,
    bpm: f32,
}

impl Synthesizer {
    /// Create a new synthesizer
    pub fn new(sample_rate: f32) -> Self {
        Self {
            oscillators: Vec::new(),
            filters: Vec::new(),
            envelope: Envelope::new(0.1, 0.2, 0.7, 0.3, sample_rate),
            sample_rate,
            bpm: 120.0,
        }
    }

    /// Add an oscillator to the synthesizer
    pub fn add_oscillator(&mut self, oscillator: Oscillator) {
        self.oscillators.push(oscillator);
    }

    /// Add a filter to the synthesizer
    pub fn add_filter(&mut self, filter: Filter) {
        self.filters.push(filter);
    }

    /// Set tempo in beats per minute
    pub fn set_bpm(&mut self, bpm: f32) {
        self.bpm = bpm;
    }

    /// Generate audio for a note
    pub fn play_note(&mut self, frequency: f32, duration_seconds: f32) -> AudioBuffer {
        let mut buffer = AudioBuffer::new(self.sample_rate, 1, duration_seconds);
        
        // Trigger envelope
        self.envelope.trigger();
        
        // Generate oscillator signals
        for oscillator in &mut self.oscillators {
            oscillator.set_frequency(frequency, 0.0);
            oscillator.generate(buffer.channel_mut(0), self.sample_rate);
        }
        
        // Apply envelope
        self.envelope.generate(buffer.channel_mut(0));
        
        // Apply filters
        for filter in &mut self.filters {
            filter.process(buffer.channel_mut(0));
        }
        
        buffer
    }

    /// Generate audio from EEG data
    pub fn eeg_to_audio(&mut self, eeg_data: &[f32]) -> AudioBuffer {
        // Convert EEG frequency bands to musical parameters
        let alpha_power = self.extract_band_power(eeg_data, 8.0, 13.0);
        let beta_power = self.extract_band_power(eeg_data, 13.0, 30.0);
        let theta_power = self.extract_band_power(eeg_data, 4.0, 8.0);
        
        // Map EEG data to musical parameters
        let base_frequency = 110.0 + (alpha_power * 220.0); // A2 to A3 range
        let detune = theta_power * 10.0; // +/- 10 cents
        let duration = 0.5 + (beta_power * 2.0); // 0.5 to 2.5 seconds
        
        // Set oscillator parameters based on EEG
        for oscillator in &mut self.oscillators {
            oscillator.set_frequency(base_frequency, detune);
        }
        
        // Set envelope parameters based on EEG
        let attack = 0.05 + (theta_power * 0.2);
        let decay = 0.1 + (alpha_power * 0.3);
        let sustain = 0.5 + (beta_power * 0.5);
        let release = 0.1 + (theta_power * 0.4);
        
        self.envelope = Envelope::new(attack, decay, sustain, release, self.sample_rate);
        
        // Generate audio
        self.play_note(base_frequency, duration)
    }

    /// Extract power in a specific frequency band
    fn extract_band_power(&self, data: &[f32], low_freq: f32, high_freq: f32) -> f32 {
        // Simple implementation - in a real system, you'd use FFT
        // For now, we'll simulate band power extraction
        if data.is_empty() {
            return 0.0;
        }
        
        // Calculate indices for frequency band
        let low_idx = (low_freq * data.len() as f32 / self.sample_rate) as usize;
        let high_idx = (high_freq * data.len() as f32 / self.sample_rate) as usize;
        
        // Clamp indices
        let low_idx = low_idx.min(data.len() - 1);
        let high_idx = high_idx.min(data.len() - 1).max(low_idx);
        
        // Calculate mean power in band
        let band_data = &data[low_idx..=high_idx];
        let power: f32 = band_data.iter().map(|&x| x * x).sum();
        power / band_data.len() as f32
    }
}

/// Audio effect processors
pub mod effects {
    use super::*;
    
    /// Reverb effect
    pub struct Reverb {
        delay_lines: Vec<Vec<Sample>>,
        feedback: f32,
        mix: f32,
    }
    
    impl Reverb {
        pub fn new(feedback: f32, mix: f32) -> Self {
            // Create multiple delay lines for stereo reverb
            let delay_lines = vec![vec![0.0; 44100], vec![0.0; 44100]]; // 1 second at 44.1kHz
            Self {
                delay_lines,
                feedback,
                mix,
            }
        }
        
        pub fn process(&mut self, buffer: &mut AudioBuffer) {
            // Simple all-pass filter reverb implementation
            for channel in 0..buffer.channels {
                let channel_data = buffer.channel_mut(channel);
                for sample in channel_data.iter_mut() {
                    // Apply reverb effect
                    let wet = *sample * self.feedback;
                    *sample = *sample * (1.0 - self.mix) + wet * self.mix;
                }
            }
        }
    }
    
    /// Distortion effect
    pub struct Distortion {
        drive: f32,
        mix: f32,
    }
    
    impl Distortion {
        pub fn new(drive: f32, mix: f32) -> Self {
            Self { drive, mix }
        }
        
        pub fn process(&mut self, buffer: &mut AudioBuffer) {
            for channel in 0..buffer.channels {
                let channel_data = buffer.channel_mut(channel);
                for sample in channel_data.iter_mut() {
                    // Apply distortion (tanh clipping)
                    let driven = *sample * self.drive;
                    let distorted = driven.tanh();
                    *sample = *sample * (1.0 - self.mix) + distorted * self.mix;
                }
            }
        }
    }
    
    /// Chorus effect
    pub struct Chorus {
        delay: f32,
        depth: f32,
        rate: f32,
        mix: f32,
        sample_rate: f32,
        phase: f32,
    }
    
    impl Chorus {
        pub fn new(delay: f32, depth: f32, rate: f32, mix: f32, sample_rate: f32) -> Self {
            Self {
                delay,
                depth,
                rate,
                mix,
                sample_rate,
                phase: 0.0,
            }
        }
        
        pub fn process(&mut self, buffer: &mut AudioBuffer) {
            let delay_samples = (self.delay * self.sample_rate) as usize;
            let depth_samples = (self.depth * self.sample_rate) as usize;
            
            for channel in 0..buffer.channels {
                let channel_data = buffer.channel_mut(channel);
                for (i, sample) in channel_data.iter_mut().enumerate() {
                    // Modulate delay time with LFO
                    let lfo = (self.phase + i as f32 / buffer.samples_per_channel() as f32 * 2.0 * PI).sin();
                    let mod_delay = delay_samples + (lfo * depth_samples as f32) as usize;
                    
                    // Read delayed sample (simple linear interpolation)
                    let delay_idx = i.saturating_sub(mod_delay);
                    if delay_idx < channel_data.len() {
                        let delayed = channel_data[delay_idx];
                        *sample = *sample * (1.0 - self.mix) + delayed * self.mix;
                    }
                }
            }
            
            self.phase += 2.0 * PI * self.rate / self.sample_rate;
            if self.phase >= 2.0 * PI {
                self.phase -= 2.0 * PI;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer_creation() {
        let buffer = AudioBuffer::new(44100.0, 2, 1.0);
        assert_eq!(buffer.sample_rate, 44100.0);
        assert_eq!(buffer.channels, 2);
        assert_eq!(buffer.samples.len(), 88200); // 44100 * 2 channels * 1 second
    }

    #[test]
    fn test_oscillator_sine() {
        let mut oscillator = Oscillator::new(Waveform::Sine, 440.0, 1.0);
        let mut buffer = vec![0.0; 100];
        oscillator.generate(&mut buffer, 44100.0);
        
        // First sample should be near 0 (sin(0) = 0)
        assert!((buffer[0] - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_oscillator_square() {
        let mut oscillator = Oscillator::new(Waveform::Square, 440.0, 1.0);
        let mut buffer = vec![0.0; 100];
        oscillator.generate(&mut buffer, 44100.0);
        
        // First sample should be 1.0 (positive cycle)
        assert_eq!(buffer[0], 1.0);
    }

    #[test]
    fn test_envelope_generation() {
        let mut envelope = Envelope::new(0.1, 0.1, 0.7, 0.1, 44100.0);
        let mut buffer = vec![1.0; 1000];
        envelope.trigger();
        envelope.generate(&mut buffer);
        
        // Buffer should not be empty and should have values between 0 and 1
        assert!(!buffer.is_empty());
        assert!(buffer.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_synthesizer_creation() {
        let synth = Synthesizer::new(44100.0);
        assert_eq!(synth.sample_rate, 44100.0);
    }

    #[test]
    fn test_effects_creation() {
        let _reverb = effects::Reverb::new(0.5, 0.3);
        let _distortion = effects::Distortion::new(2.0, 0.5);
        let _chorus = effects::Chorus::new(0.01, 0.005, 0.5, 0.3, 44100.0);
    }
}