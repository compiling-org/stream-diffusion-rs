//! Simplified Fractal Shader Renderer for Stream Diffusion
//!
//! This module provides basic fractal visualization capabilities for the Stream Diffusion
//! frontend, adapted from Neuro-Emotive AI patterns but simplified for creative/research use.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Fractal shader types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FractalType {
    Mandelbrot,
    Julia,
    BurningShip,
}

/// Fractal shader parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalParameters {
    pub fractal_type: FractalType,
    pub iterations: u32,
    pub zoom: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    pub rotation: f32,
    pub hue_shift: f32,
    pub saturation: f32,
    pub brightness: f32,
    pub animation_speed: f32,
}

impl Default for FractalParameters {
    fn default() -> Self {
        Self {
            fractal_type: FractalType::Mandelbrot,
            iterations: 100,
            zoom: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
            rotation: 0.0,
            hue_shift: 0.0,
            saturation: 1.0,
            brightness: 1.0,
            animation_speed: 0.5,
        }
    }
}

/// Fractal shader renderer
pub struct FractalShaderRenderer {
    parameters: FractalParameters,
    time: f32,
}

impl FractalShaderRenderer {
    pub fn new() -> Self {
        Self {
            parameters: FractalParameters::default(),
            time: 0.0,
        }
    }

    pub fn with_parameters(parameters: FractalParameters) -> Self {
        Self {
            parameters,
            time: 0.0,
        }
    }

    /// Update shader parameters
    pub fn update_parameters(&mut self, params: FractalParameters) {
        self.parameters = params;
    }

    /// Update animation time
    pub fn update_time(&mut self, delta_time: f32) {
        self.time += delta_time * self.parameters.animation_speed;
    }

    /// Generate vertex shader source
    pub fn vertex_shader_source() -> &'static str {
        r#"
        attribute vec2 a_position;
        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
        }
        "#
    }

    /// Generate fragment shader source based on fractal type
    pub fn fragment_shader_source(&self) -> String {
        let fractal_function = match self.parameters.fractal_type {
            FractalType::Mandelbrot => Self::mandelbrot_function(),
            FractalType::Julia => Self::julia_function(),
            FractalType::BurningShip => Self::burning_ship_function(),
        };

        format!(r#"
        precision mediump float;
        uniform vec2 u_resolution;
        uniform float u_time;
        uniform float u_iterations;
        uniform float u_zoom;
        uniform vec2 u_offset;
        uniform float u_rotation;
        uniform float u_hue_shift;
        uniform float u_saturation;
        uniform float u_brightness;

        // HSV to RGB conversion
        vec3 hsv2rgb(vec3 c) {{
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }}

        // Complex number operations
        vec2 complex_mul(vec2 a, vec2 b) {{
            return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }}

        vec2 complex_square(vec2 z) {{
            return vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
        }}

        float complex_mag(vec2 z) {{
            return sqrt(z.x * z.x + z.y * z.y);
        }}

        {}

        void main() {{
            vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
            uv *= u_zoom;

            // Apply rotation
            float cos_rot = cos(u_rotation + u_time * 0.1);
            float sin_rot = sin(u_rotation + u_time * 0.1);
            uv = vec2(uv.x * cos_rot - uv.y * sin_rot, uv.x * sin_rot + uv.y * cos_rot);

            // Apply offset
            uv += u_offset;

            float fractal_value = {};

            // Color mapping
            float hue = fract(fractal_value * 0.1 + u_hue_shift + u_time * 0.05);
            vec3 color = hsv2rgb(vec3(hue, u_saturation, u_brightness));

            // Add some animation effects
            color += sin(u_time + fractal_value * 10.0) * 0.1;

            gl_FragColor = vec4(color, 1.0);
        }}
        "#, fractal_function, self.fractal_call())
    }

    fn mandelbrot_function() -> &'static str {
        r#"
        float mandelbrot(vec2 c) {
            vec2 z = vec2(0.0, 0.0);
            float iterations = 0.0;

            for (float i = 0.0; i < 200.0; i++) {
                if (i >= u_iterations) break;
                if (complex_mag(z) > 2.0) break;

                z = complex_square(z) + c;
                iterations = i;
            }

            return iterations / u_iterations;
        }
        "#
    }

    fn julia_function() -> &'static str {
        r#"
        float julia(vec2 z) {
            vec2 c = vec2(-0.7 + 0.3 * sin(u_time * 0.5), 0.3 * cos(u_time * 0.3));
            float iterations = 0.0;

            for (float i = 0.0; i < 200.0; i++) {
                if (i >= u_iterations) break;
                if (complex_mag(z) > 2.0) break;

                z = complex_square(z) + c;
                iterations = i;
            }

            return iterations / u_iterations;
        }
        "#
    }

    fn burning_ship_function() -> &'static str {
        r#"
        float burning_ship(vec2 c) {
            vec2 z = vec2(0.0, 0.0);
            float iterations = 0.0;

            for (float i = 0.0; i < 200.0; i++) {
                if (i >= u_iterations) break;
                if (complex_mag(z) > 2.0) break;

                z = vec2(abs(z.x), abs(z.y));
                z = complex_square(z) + c;
                iterations = i;
            }

            return iterations / u_iterations;
        }
        "#
    }

    fn fractal_call(&self) -> &'static str {
        match self.parameters.fractal_type {
            FractalType::Mandelbrot => "mandelbrot(uv)",
            FractalType::Julia => "julia(uv)",
            FractalType::BurningShip => "burning_ship(uv)",
        }
    }

    /// Get current parameters
    pub fn parameters(&self) -> &FractalParameters {
        &self.parameters
    }

    /// Get current time
    pub fn time(&self) -> f32 {
        self.time
    }
}

/// Preset configurations for different fractal types
pub struct FractalPresets;

impl FractalPresets {
    pub fn mandelbrot() -> FractalParameters {
        FractalParameters {
            fractal_type: FractalType::Mandelbrot,
            iterations: 100,
            zoom: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
            rotation: 0.0,
            hue_shift: 0.0,
            saturation: 1.0,
            brightness: 1.0,
            animation_speed: 0.5,
        }
    }

    pub fn julia() -> FractalParameters {
        FractalParameters {
            fractal_type: FractalType::Julia,
            iterations: 150,
            zoom: 1.5,
            offset_x: 0.0,
            offset_y: 0.0,
            rotation: 0.0,
            hue_shift: 0.3,
            saturation: 0.8,
            brightness: 1.2,
            animation_speed: 0.3,
        }
    }

    pub fn burning_ship() -> FractalParameters {
        FractalParameters {
            fractal_type: FractalType::BurningShip,
            iterations: 80,
            zoom: 0.8,
            offset_x: 0.0,
            offset_y: 0.0,
            rotation: 0.0,
            hue_shift: 0.6,
            saturation: 1.0,
            brightness: 0.9,
            animation_speed: 0.2,
        }
    }

    pub fn creative_flow() -> FractalParameters {
        FractalParameters {
            fractal_type: FractalType::Julia,
            iterations: 120,
            zoom: 2.0,
            offset_x: 0.1,
            offset_y: -0.1,
            rotation: 0.5,
            hue_shift: 0.8,
            saturation: 0.9,
            brightness: 1.1,
            animation_speed: 0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractal_renderer_creation() {
        let renderer = FractalShaderRenderer::new();
        assert_eq!(renderer.parameters().iterations, 100);
        assert_eq!(renderer.time(), 0.0);
    }

    #[test]
    fn test_parameter_update() {
        let mut renderer = FractalShaderRenderer::new();
        let new_params = FractalParameters {
            iterations: 200,
            ..Default::default()
        };

        renderer.update_parameters(new_params);
        assert_eq!(renderer.parameters().iterations, 200);
    }

    #[test]
    fn test_time_update() {
        let mut renderer = FractalShaderRenderer::new();
        renderer.update_time(1.0);
        assert_eq!(renderer.time(), 0.5); // animation_speed is 0.5
    }

    #[test]
    fn test_presets() {
        let mandelbrot = FractalPresets::mandelbrot();
        assert_eq!(mandelbrot.iterations, 100);

        let julia = FractalPresets::julia();
        assert_eq!(julia.iterations, 150);

        let burning_ship = FractalPresets::burning_ship();
        assert_eq!(burning_ship.iterations, 80);
    }

    #[test]
    fn test_vertex_shader_source() {
        let source = FractalShaderRenderer::vertex_shader_source();
        assert!(source.contains("attribute vec2 a_position"));
        assert!(source.contains("gl_Position"));
    }

    #[test]
    fn test_fragment_shader_source() {
        let renderer = FractalShaderRenderer::new();
        let source = renderer.fragment_shader_source();
        assert!(source.contains("precision mediump float"));
        assert!(source.contains("mandelbrot"));
        assert!(source.contains("hsv2rgb"));
    }
}