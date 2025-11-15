//! Real-time shader animations for visual effects
//!
//! This module provides implementations for real-time shader-based animations
//! using WebGL-compatible GLSL shaders for fractal rendering, particle systems,
//! and other visual effects.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;

/// Shader animation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderParams {
    pub time: f32,
    pub resolution: [f32; 2],
    pub mouse: [f32; 2],
    pub uniforms: HashMap<String, f32>,
}

/// Fractal shader types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FractalType {
    Mandelbrot,
    Julia { cx: f32, cy: f32 },
    BurningShip,
    Newton { power: f32 },
}

/// Particle system for shader animations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Particle {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub color: [f32; 4],
    pub lifetime: f32,
    pub max_lifetime: f32,
}

/// Shader animation engine
pub struct ShaderAnimationEngine {
    params: ShaderParams,
    particles: Vec<Particle>,
    fractal_type: FractalType,
    time_accumulator: f32,
}

impl ShaderAnimationEngine {
    /// Create a new shader animation engine
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            params: ShaderParams {
                time: 0.0,
                resolution: [width, height],
                mouse: [width / 2.0, height / 2.0],
                uniforms: HashMap::new(),
            },
            particles: Vec::new(),
            fractal_type: FractalType::Mandelbrot,
            time_accumulator: 0.0,
        }
    }

    /// Update animation parameters
    pub fn update(&mut self, delta_time: f32) {
        self.params.time += delta_time;
        self.time_accumulator += delta_time;
        
        // Update particles
        self.update_particles(delta_time);
        
        // Add new particles periodically
        if self.time_accumulator > 0.1 {
            self.add_particles(5);
            self.time_accumulator = 0.0;
        }
    }

    /// Update particle system
    fn update_particles(&mut self, delta_time: f32) {
        // Update existing particles
        self.particles.retain_mut(|particle| {
            particle.position[0] += particle.velocity[0] * delta_time;
            particle.position[1] += particle.velocity[1] * delta_time;
            particle.lifetime -= delta_time;
            
            // Fade out particles as they age
            let alpha_factor = particle.lifetime / particle.max_lifetime;
            particle.color[3] = alpha_factor;
            
            particle.lifetime > 0.0
        });
    }

    /// Add new particles to the system
    fn add_particles(&mut self, count: usize) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..count {
            let angle = rng.gen::<f32>() * 2.0 * PI;
            let speed = rng.gen::<f32>() * 100.0 + 50.0;
            let lifetime = rng.gen::<f32>() * 3.0 + 2.0;
            
            let particle = Particle {
                position: [self.params.mouse[0], self.params.mouse[1]],
                velocity: [angle.cos() * speed, angle.sin() * speed],
                color: [
                    rng.gen::<f32>(),
                    rng.gen::<f32>(),
                    rng.gen::<f32>(),
                    1.0,
                ],
                lifetime,
                max_lifetime: lifetime,
            };
            
            self.particles.push(particle);
        }
    }

    /// Set mouse position
    pub fn set_mouse_position(&mut self, x: f32, y: f32) {
        self.params.mouse = [x, y];
    }

    /// Set fractal type
    pub fn set_fractal_type(&mut self, fractal_type: FractalType) {
        self.fractal_type = fractal_type;
    }

    /// Add uniform parameter
    pub fn set_uniform(&mut self, name: String, value: f32) {
        self.params.uniforms.insert(name, value);
    }

    /// Generate fragment shader code for fractal rendering
    pub fn generate_fractal_shader(&self) -> String {
        let shader_code = match &self.fractal_type {
            FractalType::Mandelbrot => self.generate_mandelbrot_shader(),
            FractalType::Julia { cx, cy } => self.generate_julia_shader(*cx, *cy),
            FractalType::BurningShip => self.generate_burning_ship_shader(),
            FractalType::Newton { power } => self.generate_newton_shader(*power),
        };
        
        shader_code
    }

    /// Generate Mandelbrot fractal shader
    fn generate_mandelbrot_shader(&self) -> String {
        format!(
            r#"#version 300 es
precision highp float;

uniform float time;
uniform vec2 resolution;
out vec4 fragColor;

void main() {{
    vec2 c = (gl_FragCoord.xy - 0.5 * resolution) / min(resolution.x, resolution.y);
    c.x += sin(time * 0.1) * 0.1;
    c.y += cos(time * 0.1) * 0.1;
    
    vec2 z = vec2(0.0);
    float iterations = 0.0;
    const float max_iterations = 100.0;
    
    for (float i = 0.0; i < max_iterations; i++) {{
        if (dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iterations++;
    }}
    
    float intensity = iterations / max_iterations;
    vec3 color = 0.5 + 0.5 * cos(6.28318 * (vec3(0.0, 0.33, 0.67) + intensity + time * 0.1));
    fragColor = vec4(color * (1.0 - intensity), 1.0);
}}"#
        )
    }

    /// Generate Julia fractal shader
    fn generate_julia_shader(&self, cx: f32, cy: f32) -> String {
        format!(
            r#"#version 300 es
precision highp float;

uniform float time;
uniform vec2 resolution;
out vec4 fragColor;

void main() {{
    vec2 c = vec2({}, {});
    vec2 z = (gl_FragCoord.xy - 0.5 * resolution) / min(resolution.x, resolution.y);
    z.x += sin(time * 0.2) * 0.2;
    z.y += cos(time * 0.2) * 0.2;
    
    float iterations = 0.0;
    const float max_iterations = 100.0;
    
    for (float i = 0.0; i < max_iterations; i++) {{
        if (dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iterations++;
    }}
    
    float intensity = iterations / max_iterations;
    vec3 color = 0.5 + 0.5 * cos(6.28318 * (vec3(0.67, 0.0, 0.33) + intensity + time * 0.15));
    fragColor = vec4(color * intensity, 1.0);
}}"#,
            cx, cy
        )
    }

    /// Generate Burning Ship fractal shader
    fn generate_burning_ship_shader(&self) -> String {
        format!(
            r#"#version 300 es
precision highp float;

uniform float time;
uniform vec2 resolution;
out vec4 fragColor;

void main() {{
    vec2 c = (gl_FragCoord.xy - 0.5 * resolution) / min(resolution.x, resolution.y);
    c.x += sin(time * 0.15) * 0.15;
    c.y += cos(time * 0.15) * 0.15;
    
    vec2 z = vec2(0.0);
    float iterations = 0.0;
    const float max_iterations = 100.0;
    
    for (float i = 0.0; i < max_iterations; i++) {{
        if (dot(z, z) > 4.0) break;
        z = vec2(abs(z.x), abs(z.y));
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iterations++;
    }}
    
    float intensity = iterations / max_iterations;
    vec3 color = 0.5 + 0.5 * cos(6.28318 * (vec3(0.33, 0.67, 0.0) + intensity + time * 0.12));
    fragColor = vec4(color * (1.0 - intensity), 1.0);
}}"#
        )
    }

    /// Generate Newton fractal shader
    fn generate_newton_shader(&self, power: f32) -> String {
        format!(
            r#"#version 300 es
precision highp float;

uniform float time;
uniform vec2 resolution;
out vec4 fragColor;

void main() {{
    vec2 z = (gl_FragCoord.xy - 0.5 * resolution) / min(resolution.x, resolution.y) * 2.0;
    z.x += sin(time * 0.08) * 0.2;
    z.y += cos(time * 0.08) * 0.2;
    
    float iterations = 0.0;
    const float max_iterations = 100.0;
    const float epsilon = 0.001;
    
    for (float i = 0.0; i < max_iterations; i++) {{
        if (abs(length(z) - 1.0) < epsilon) break;
        
        float r = length(z);
        float theta = atan(z.y, z.x);
        float new_r = pow(r, {});
        float new_theta = theta * {};
        
        z = vec2(new_r * cos(new_theta), new_r * sin(new_theta));
        z = (z * ({} - 1.0) + vec2(1.0) / pow(length(z), {} - 2.0)) / {};
        
        iterations++;
    }}
    
    float intensity = iterations / max_iterations;
    vec3 color = 0.5 + 0.5 * cos(6.28318 * (vec3(0.0, 0.67, 0.33) + intensity + time * 0.09));
    fragColor = vec4(color * intensity, 1.0);
}}"#,
            power, power, power, power, power
        )
    }

    /// Generate particle system shader
    pub fn generate_particle_shader(&self) -> String {
        format!(
            r#"#version 300 es
precision highp float;

uniform float time;
uniform vec2 resolution;
uniform vec2 mouse;
out vec4 fragColor;

// Particle data would be passed as uniforms or textures in a real implementation
// For this example, we'll generate procedural particles

void main() {{
    vec2 uv = gl_FragCoord.xy / resolution;
    vec3 color = vec3(0.0);
    
    // Generate procedural particles
    for (int i = 0; i < 50; i++) {{
        float index = float(i);
        float particle_time = time + index * 0.1;
        vec2 center = vec2(
            0.5 + sin(particle_time * 0.5 + index) * 0.3,
            0.5 + cos(particle_time * 0.3 + index) * 0.3
        );
        
        vec2 diff = uv - center;
        float dist = length(diff);
        float size = 0.02 + sin(particle_time) * 0.01;
        
        if (dist < size) {{
            float alpha = 1.0 - smoothstep(0.0, size, dist);
            vec3 particle_color = 0.5 + 0.5 * cos(vec3(0.0, 2.0, 4.0) + particle_time);
            color += particle_color * alpha;
        }}
    }}
    
    fragColor = vec4(color, 1.0);
}}"#
        )
    }

    /// Generate audio-reactive shader
    pub fn generate_audio_shader(&self, frequencies: &[f32]) -> String {
        // Convert frequencies to a string representation
        let freq_str: String = frequencies
            .iter()
            .take(8) // Limit to first 8 frequencies
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        
        format!(
            r#"#version 300 es
precision highp float;

uniform float time;
uniform vec2 resolution;
uniform float frequencies[8];
out vec4 fragColor;

void main() {{
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution) / min(resolution.x, resolution.y);
    
    // Base pattern
    float pattern = sin(uv.x * 10.0 + time) * cos(uv.y * 10.0 + time);
    
    // Audio-reactive deformation
    float audio_deform = 0.0;
    for (int i = 0; i < 8; i++) {{
        audio_deform += frequencies[i] * sin(float(i) * 0.5 + time * 2.0);
    }}
    
    uv.x += audio_deform * 0.1;
    uv.y += audio_deform * 0.05;
    
    // Color based on audio
    float r = 0.5 + 0.5 * sin(time + uv.x * 2.0 + audio_deform);
    float g = 0.5 + 0.5 * sin(time + uv.y * 3.0 + audio_deform * 2.0);
    float b = 0.5 + 0.5 * sin(time + (uv.x + uv.y) + audio_deform * 3.0);
    
    fragColor = vec4(r, g, b, 1.0);
}}"#
        )
    }

    /// Get current particle data for rendering
    pub fn get_particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Get shader parameters
    pub fn get_params(&self) -> &ShaderParams {
        &self.params
    }
}

/// Audio synthesis for shader animations
pub struct AudioSynthesizer {
    sample_rate: f32,
    time: f32,
    oscillators: Vec<Oscillator>,
}

/// Audio oscillator for sound generation
#[derive(Debug, Clone)]
pub struct Oscillator {
    pub waveform: Waveform,
    pub frequency: f32,
    pub amplitude: f32,
    pub phase: f32,
}

/// Waveform types for audio synthesis
#[derive(Debug, Clone)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
    Noise,
}

impl AudioSynthesizer {
    /// Create a new audio synthesizer
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            time: 0.0,
            oscillators: Vec::new(),
        }
    }

    /// Add an oscillator to the synthesizer
    pub fn add_oscillator(&mut self, oscillator: Oscillator) {
        self.oscillators.push(oscillator);
    }

    /// Generate audio samples
    pub fn generate_samples(&mut self, num_samples: usize) -> Vec<f32> {
        let mut samples = Vec::with_capacity(num_samples);
        
        for _ in 0..num_samples {
            let mut sample = 0.0;
            
            for oscillator in &mut self.oscillators {
                sample += oscillator.generate_sample(self.time);
                oscillator.phase += oscillator.frequency / self.sample_rate;
                // Keep phase in [0, 1] range
                oscillator.phase -= oscillator.phase.floor();
            }
            
            // Clamp sample to [-1, 1] range
            sample = sample.clamp(-1.0, 1.0);
            samples.push(sample);
            
            self.time += 1.0 / self.sample_rate;
        }
        
        samples
    }

    /// Generate frequency spectrum for audio-reactive visuals
    pub fn generate_spectrum(&self, num_bands: usize) -> Vec<f32> {
        // Simple implementation - in a real system, you'd use FFT
        let mut spectrum = vec![0.0; num_bands];
        
        for (i, oscillator) in self.oscillators.iter().enumerate() {
            let band = (oscillator.frequency / 100.0) as usize % num_bands;
            spectrum[band] += oscillator.amplitude;
        }
        
        // Normalize spectrum
        let max_val = spectrum.iter().fold(0.0, |a, &b| a.max(b));
        if max_val > 0.0 {
            for val in &mut spectrum {
                *val /= max_val;
            }
        }
        
        spectrum
    }
}

impl Oscillator {
    /// Generate a single sample
    pub fn generate_sample(&self, time: f32) -> f32 {
        let phase_rad = self.phase * 2.0 * PI;
        
        let sample = match self.waveform {
            Waveform::Sine => phase_rad.sin(),
            Waveform::Square => {
                if phase_rad % (2.0 * PI) < PI {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Sawtooth => 2.0 * (self.phase - self.phase.floor()) - 1.0,
            Waveform::Triangle => {
                let t = self.phase - self.phase.floor();
                if t < 0.5 {
                    4.0 * t - 1.0
                } else {
                    3.0 - 4.0 * t
                }
            }
            Waveform::Noise => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                rng.gen::<f32>() * 2.0 - 1.0
            }
        };
        
        sample * self.amplitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_animation_engine_creation() {
        let engine = ShaderAnimationEngine::new(800.0, 600.0);
        assert_eq!(engine.get_params().resolution, [800.0, 600.0]);
    }

    #[test]
    fn test_shader_update() {
        let mut engine = ShaderAnimationEngine::new(800.0, 600.0);
        let initial_time = engine.get_params().time;
        engine.update(0.1);
        assert_eq!(engine.get_params().time, initial_time + 0.1);
    }

    #[test]
    fn test_fractal_shader_generation() {
        let engine = ShaderAnimationEngine::new(800.0, 600.0);
        let shader_code = engine.generate_fractal_shader();
        assert!(!shader_code.is_empty());
        assert!(shader_code.contains("void main()"));
    }

    #[test]
    fn test_audio_synthesizer() {
        let mut synth = AudioSynthesizer::new(44100.0);
        synth.add_oscillator(Oscillator {
            waveform: Waveform::Sine,
            frequency: 440.0,
            amplitude: 0.5,
            phase: 0.0,
        });
        
        let samples = synth.generate_samples(100);
        assert_eq!(samples.len(), 100);
        
        let spectrum = synth.generate_spectrum(8);
        assert_eq!(spectrum.len(), 8);
    }

    #[test]
    fn test_oscillator_generation() {
        let oscillator = Oscillator {
            waveform: Waveform::Sine,
            frequency: 440.0,
            amplitude: 1.0,
            phase: 0.0,
        };
        
        let sample = oscillator.generate_sample(0.0);
        assert!((sample - 0.0).abs() < 0.001); // sin(0) = 0
    }
}