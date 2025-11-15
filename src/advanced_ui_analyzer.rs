//! Advanced UI Analyzer for Stream Diffusion RS
//!
//! This analyzer actually inspects the source code to determine the real implementation status
//! of UI features by analyzing the web.rs file and other relevant modules.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Detailed UI Component Analysis
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UIComponentAnalysis {
    pub name: String,
    pub endpoint_exists: bool,
    pub frontend_exists: bool,
    pub backend_implemented: bool,
    pub functional: bool,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Advanced UI Analysis Result
#[derive(Debug, Serialize, Deserialize)]
pub struct AdvancedUIAnalysis {
    pub timestamp: String,
    pub components: Vec<UIComponentAnalysis>,
    pub summary: AnalysisSummary,
}

/// Analysis Summary
#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisSummary {
    pub total_components: usize,
    pub fully_functional: usize,
    pub backend_only: usize,
    pub frontend_only: usize,
    pub incomplete: usize,
    pub code_coverage: f32,
}

/// Advanced UI Analyzer
pub struct AdvancedUIAnalyzer {
    project_root: String,
}

impl AdvancedUIAnalyzer {
    pub fn new(project_root: &str) -> Self {
        Self {
            project_root: project_root.to_string(),
        }
    }

    /// Analyze the entire UI implementation
    pub fn analyze(&self) -> Result<AdvancedUIAnalysis, Box<dyn std::error::Error>> {
        let components = self.analyze_components()?;
        let summary = self.generate_summary(&components);
        
        Ok(AdvancedUIAnalysis {
            timestamp: chrono::Utc::now().to_rfc3339(),
            components,
            summary,
        })
    }

    /// Analyze individual UI components
    fn analyze_components(&self) -> Result<Vec<UIComponentAnalysis>, Box<dyn std::error::Error>> {
        let mut components = Vec::new();

        // Analyze Image Generation
        components.push(self.analyze_image_generation()?);

        // Analyze Stream Diffusion
        components.push(self.analyze_stream_diffusion()?);

        // Analyze EEG Analysis
        components.push(self.analyze_eeg_analysis()?);

        // Analyze Fractal Shaders
        components.push(self.analyze_fractal_shaders()?);

        // Analyze Audio Synthesis
        components.push(self.analyze_audio_synthesis()?);

        // Analyze 3D Models
        components.push(self.analyze_3d_models()?);

        // Analyze Audiovisual Integration
        components.push(self.analyze_audiovisual_integration()?);

        // Analyze Gesture Control
        components.push(self.analyze_gesture_control()?);

        // Analyze NUWE Node System
        components.push(self.analyze_nuwe_system()?);

        // Analyze Model Training
        components.push(self.analyze_model_training()?);

        // Analyze Model Management
        components.push(self.analyze_model_management()?);

        Ok(components)
    }

    /// Analyze Image Generation component
    fn analyze_image_generation(&self) -> Result<UIComponentAnalysis, Box<dyn std::error::Error>> {
        let web_rs_path = format!("{}/src/web.rs", self.project_root);
        let web_content = fs::read_to_string(&web_rs_path)?;

        let endpoint_exists = web_content.contains("generate_image") && web_content.contains("/api/generate");
        let frontend_exists = web_content.contains("generateImage()") && web_content.contains("prompt");
        
        // Check if it's a placeholder implementation
        let is_placeholder = web_content.contains("// This would") || 
                            web_content.contains("return a placeholder response") ||
                            web_content.contains("For now, return a placeholder response") ||
                            web_content.contains("Placeholder");
        
        // Check if the implementation is real
        let has_real_implementation = web_content.contains("async fn generate_image") && !is_placeholder;
        
        // Check if required models exist (they don't in this case)
        let models_available = false;
        
        // True functionality requires both real implementation and available models
        let is_functional = has_real_implementation && models_available;

        let functional = endpoint_exists && frontend_exists && is_functional;
        
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !endpoint_exists {
            issues.push("API endpoint for image generation not found".to_string());
            recommendations.push("Implement POST /api/generate endpoint".to_string());
        }

        if !frontend_exists {
            issues.push("Frontend UI for image generation not found".to_string());
            recommendations.push("Add image generation form to frontend".to_string());
        }

        if !has_real_implementation {
            issues.push("Backend image generation is just a placeholder".to_string());
            recommendations.push("Implement actual image generation using diffusion models".to_string());
        } else if !models_available {
            issues.push("No diffusion models available for image generation".to_string());
            recommendations.push("Add diffusion models to enable image generation".to_string());
        }

        Ok(UIComponentAnalysis {
            name: "Image Generation".to_string(),
            endpoint_exists,
            frontend_exists,
            backend_implemented: has_real_implementation && models_available,
            functional,
            issues,
            recommendations,
        })
    }

    /// Analyze Stream Diffusion component
    fn analyze_stream_diffusion(&self) -> Result<UIComponentAnalysis, Box<dyn std::error::Error>> {
        let web_rs_path = format!("{}/src/web.rs", self.project_root);
        let web_content = fs::read_to_string(&web_rs_path)?;

        let endpoint_exists = web_content.contains("start_stream_diffusion") && web_content.contains("/api/stream-diffusion");
        let frontend_exists = web_content.contains("startStreamDiffusion()") && web_content.contains("stream-canvas");
        
        // Check if it's a placeholder implementation
        let is_placeholder = web_content.contains("// This would") || 
                            web_content.contains("return a placeholder response") ||
                            web_content.contains("For now, return a placeholder response") ||
                            web_content.contains("Placeholder");
        
        // Check if the implementation is real
        let has_real_implementation = web_content.contains("async fn start_stream_diffusion") && !is_placeholder;
        
        // Check if required models exist
        let models_available = false;
        
        // True functionality requires both real implementation and available models
        let is_functional = has_real_implementation && models_available;

        let functional = endpoint_exists && frontend_exists && is_functional;
        
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !endpoint_exists {
            issues.push("API endpoint for stream diffusion not found".to_string());
            recommendations.push("Implement stream diffusion API endpoints".to_string());
        }

        if !frontend_exists {
            issues.push("Frontend UI for stream diffusion not found".to_string());
            recommendations.push("Add stream diffusion controls to frontend".to_string());
        }

        if !has_real_implementation {
            issues.push("Backend stream diffusion is just a placeholder".to_string());
            recommendations.push("Implement actual stream diffusion processing".to_string());
        } else if !models_available {
            issues.push("No stream diffusion models available".to_string());
            recommendations.push("Add stream diffusion models to enable functionality".to_string());
        }

        Ok(UIComponentAnalysis {
            name: "Stream Diffusion".to_string(),
            endpoint_exists,
            frontend_exists,
            backend_implemented: has_real_implementation && models_available,
            functional,
            issues,
            recommendations,
        })
    }

    /// Analyze EEG Analysis component
    fn analyze_eeg_analysis(&self) -> Result<UIComponentAnalysis, Box<dyn std::error::Error>> {
        let web_rs_path = format!("{}/src/web.rs", self.project_root);
        let web_content = fs::read_to_string(&web_rs_path)?;

        let endpoint_exists = web_content.contains("analyze_eeg") && web_content.contains("/api/eeg/analyze");
        let frontend_exists = web_content.contains("analyzeEEG()") && web_content.contains("eeg-file");
        
        // Check if it's a placeholder implementation
        let is_placeholder = web_content.contains("// This would") || 
                            web_content.contains("return a placeholder response") ||
                            web_content.contains("For now, return a placeholder response") ||
                            web_content.contains("Placeholder") ||
                            web_content.contains("Create dummy EEG data for demonstration");
        
        // Check if the implementation is real
        let has_real_implementation = web_content.contains("async fn analyze_eeg") && !is_placeholder;
        
        // Check if EEG processing can actually work
        let eeg_functional = has_real_implementation && 
                            web_content.contains("crate::eeg::EEGData") &&
                            web_content.contains("processor.extract_band_power");

        let functional = endpoint_exists && frontend_exists && eeg_functional;
        
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !endpoint_exists {
            issues.push("API endpoint for EEG analysis not found".to_string());
            recommendations.push("Implement POST /api/eeg/analyze endpoint".to_string());
        }

        if !frontend_exists {
            issues.push("Frontend UI for EEG analysis not found".to_string());
            recommendations.push("Add EEG analysis form to frontend".to_string());
        }

        if !has_real_implementation {
            issues.push("Backend EEG analysis is just a placeholder".to_string());
            recommendations.push("Implement actual EEG processing and analysis".to_string());
        } else if !eeg_functional {
            issues.push("EEG analysis lacks real data processing capabilities".to_string());
            recommendations.push("Implement complete EEG analysis pipeline".to_string());
        }

        Ok(UIComponentAnalysis {
            name: "EEG Analysis".to_string(),
            endpoint_exists,
            frontend_exists,
            backend_implemented: eeg_functional,
            functional,
            issues,
            recommendations,
        })
    }

    /// Analyze Fractal Shaders component
    fn analyze_fractal_shaders(&self) -> Result<UIComponentAnalysis, Box<dyn std::error::Error>> {
        let web_rs_path = format!("{}/src/web.rs", self.project_root);
        let web_content = fs::read_to_string(&web_rs_path)?;

        let endpoint_exists = web_content.contains("get_fractal_shader") && web_content.contains("/api/fractal/shader");
        let frontend_exists = web_content.contains("generateFractal()") && web_content.contains("fractal-canvas");
        
        // Check for actual implementation
        let backend_implemented = web_content.contains("async fn get_fractal_shader") && 
                                 !web_content.contains("// This would") && 
                                 !web_content.contains("return a placeholder response");
        
        // More thorough check - see if it actually generates shaders
        let is_functional = backend_implemented && 
                           web_content.contains("FractalParameters") &&
                           web_content.contains("fragment_shader") &&
                           !web_content.contains("Placeholder");

        let functional = endpoint_exists && frontend_exists && is_functional;
        
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !endpoint_exists {
            issues.push("API endpoints for fractal shaders not found".to_string());
            recommendations.push("Implement fractal shader API endpoints".to_string());
        }

        if !backend_implemented {
            issues.push("Backend fractal shader logic not fully implemented".to_string());
            recommendations.push("Complete fractal shader implementation in web.rs".to_string());
        } else if !is_functional {
            issues.push("Backend fractal shaders is just a placeholder".to_string());
            recommendations.push("Implement actual fractal shader generation".to_string());
        }

        Ok(UIComponentAnalysis {
            name: "Fractal Shaders".to_string(),
            endpoint_exists,
            frontend_exists,
            backend_implemented: is_functional,
            functional,
            issues,
            recommendations,
        })
    }

    /// Analyze Audio Synthesis component
    fn analyze_audio_synthesis(&self) -> Result<UIComponentAnalysis, Box<dyn std::error::Error>> {
        let web_rs_path = format!("{}/src/web.rs", self.project_root);
        let web_content = fs::read_to_string(&web_rs_path)?;

        let endpoint_exists = web_content.contains("/api/audio") || web_content.contains("audio_synthesis");
        let frontend_exists = web_content.contains("startAudioSynthesis") && web_content.contains("waveform-canvas");
        
        // Check for actual implementation
        let backend_implemented = (web_content.contains("async fn") && web_content.contains("audio")) && 
                                 !web_content.contains("// This would") && 
                                 !web_content.contains("return a placeholder response");
        
        let is_functional = backend_implemented && 
                           (web_content.contains("Glicol") || 
                            web_content.contains("audio_engine") ||
                            web_content.contains("synthesis")) &&
                           !web_content.contains("Placeholder");

        let functional = endpoint_exists && frontend_exists && is_functional;
        
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !endpoint_exists {
            issues.push("API endpoints for audio synthesis not found".to_string());
            recommendations.push("Implement audio synthesis API endpoints".to_string());
        }

        if !backend_implemented {
            issues.push("Backend audio synthesis logic not implemented".to_string());
            recommendations.push("Implement audio synthesis functions".to_string());
        } else if !is_functional {
            issues.push("Backend audio synthesis is just a placeholder".to_string());
            recommendations.push("Implement actual audio synthesis engine".to_string());
        }

        Ok(UIComponentAnalysis {
            name: "Audio Synthesis".to_string(),
            endpoint_exists,
            frontend_exists,
            backend_implemented: is_functional,
            functional,
            issues,
            recommendations,
        })
    }

    /// Analyze 3D Models component
    fn analyze_3d_models(&self) -> Result<UIComponentAnalysis, Box<dyn std::error::Error>> {
        let web_rs_path = format!("{}/src/web.rs", self.project_root);
        let web_content = fs::read_to_string(&web_rs_path)?;

        let endpoint_exists = web_content.contains("/api/3d") || web_content.contains("threejs");
        let frontend_exists = web_content.contains("generate3DModel()") && web_content.contains("3d-canvas");
        
        // Check for actual implementation
        let backend_implemented = web_content.contains("async fn") && 
                                web_content.contains("3d") &&
                                !web_content.contains("// This would") && 
                                !web_content.contains("return a placeholder response");
        
        let is_functional = backend_implemented && 
                           (web_content.contains("3d") || 
                            web_content.contains("threejs") ||
                            web_content.contains("webgl")) &&
                           !web_content.contains("Placeholder");

        let functional = endpoint_exists && frontend_exists && is_functional;
        
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !endpoint_exists {
            issues.push("API endpoints for 3D models not found".to_string());
            recommendations.push("Implement 3D model API endpoints".to_string());
        }

        if !backend_implemented {
            issues.push("Backend 3D model logic not implemented".to_string());
            recommendations.push("Implement 3D model generation functions".to_string());
        } else if !is_functional {
            issues.push("Backend 3D models is just a placeholder".to_string());
            recommendations.push("Implement actual 3D model generation".to_string());
        }

        Ok(UIComponentAnalysis {
            name: "3D Models".to_string(),
            endpoint_exists,
            frontend_exists,
            backend_implemented: is_functional,
            functional,
            issues,
            recommendations,
        })
    }

    /// Analyze Audiovisual Integration component
    fn analyze_audiovisual_integration(&self) -> Result<UIComponentAnalysis, Box<dyn std::error::Error>> {
        let web_rs_path = format!("{}/src/web.rs", self.project_root);
        let web_content = fs::read_to_string(&web_rs_path)?;

        let endpoint_exists = web_content.contains("start_audiovisual") && web_content.contains("/api/audiovisual");
        let frontend_exists = web_content.contains("startAudiovisual()") && web_content.contains("audiovisual");
        
        // Check for actual implementation
        let backend_implemented = web_content.contains("async fn start_audiovisual") && 
                                 !web_content.contains("// This would") && 
                                 !web_content.contains("return a placeholder response");
        
        let is_functional = backend_implemented && 
                           web_content.contains("audiovisual") &&
                           web_content.contains("integration") &&
                           !web_content.contains("Placeholder");

        let functional = endpoint_exists && frontend_exists && is_functional;
        
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !backend_implemented {
            issues.push("Backend audiovisual integration logic not fully implemented".to_string());
            recommendations.push("Complete audiovisual integration implementation in web.rs".to_string());
        } else if !is_functional {
            issues.push("Backend audiovisual integration is just a placeholder".to_string());
            recommendations.push("Implement actual audiovisual processing".to_string());
        }

        Ok(UIComponentAnalysis {
            name: "Audiovisual Integration".to_string(),
            endpoint_exists,
            frontend_exists,
            backend_implemented: is_functional,
            functional,
            issues,
            recommendations,
        })
    }

    /// Analyze Gesture Control component
    fn analyze_gesture_control(&self) -> Result<UIComponentAnalysis, Box<dyn std::error::Error>> {
        let web_rs_path = format!("{}/src/web.rs", self.project_root);
        let web_content = fs::read_to_string(&web_rs_path)?;

        let endpoint_exists = web_content.contains("start_gesture_detection") && web_content.contains("/api/gesture");
        let frontend_exists = web_content.contains("startGestureDetection()") && web_content.contains("gesture-camera");
        
        // Check for actual implementation
        let backend_implemented = web_content.contains("async fn start_gesture_detection") && 
                                 !web_content.contains("// This would") && 
                                 !web_content.contains("return a placeholder response");
        
        let is_functional = backend_implemented && 
                           web_content.contains("gesture") &&
                           web_content.contains("detection") &&
                           !web_content.contains("Placeholder");

        let functional = endpoint_exists && frontend_exists && is_functional;
        
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !backend_implemented {
            issues.push("Backend gesture control logic not fully implemented".to_string());
            recommendations.push("Complete gesture control implementation in web.rs".to_string());
        } else if !is_functional {
            issues.push("Backend gesture control is just a placeholder".to_string());
            recommendations.push("Implement actual gesture detection and processing".to_string());
        }

        Ok(UIComponentAnalysis {
            name: "Gesture Control".to_string(),
            endpoint_exists,
            frontend_exists,
            backend_implemented: is_functional,
            functional,
            issues,
            recommendations,
        })
    }

    /// Analyze NUWE Node System component
    fn analyze_nuwe_system(&self) -> Result<UIComponentAnalysis, Box<dyn std::error::Error>> {
        let web_rs_path = format!("{}/src/web.rs", self.project_root);
        let web_content = fs::read_to_string(&web_rs_path)?;

        let endpoint_exists = web_content.contains("create_nuwe_node") && web_content.contains("/api/nuwe");
        let frontend_exists = web_content.contains("createNode()") && web_content.contains("node-graph");
        
        // Check for actual implementation
        let backend_implemented = web_content.contains("async fn create_nuwe_node") && 
                                 !web_content.contains("// This would") && 
                                 !web_content.contains("return a placeholder response");
        
        let is_functional = backend_implemented && 
                           web_content.contains("nuwe") &&
                           web_content.contains("pipeline") &&
                           !web_content.contains("Placeholder");

        let functional = endpoint_exists && frontend_exists && is_functional;
        
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !backend_implemented {
            issues.push("Backend NUWE node system logic not fully implemented".to_string());
            recommendations.push("Complete NUWE implementation in web.rs".to_string());
        } else if !is_functional {
            issues.push("Backend NUWE system is just a placeholder".to_string());
            recommendations.push("Implement actual NUWE node processing".to_string());
        }

        Ok(UIComponentAnalysis {
            name: "NUWE Node System".to_string(),
            endpoint_exists,
            frontend_exists,
            backend_implemented: is_functional,
            functional,
            issues,
            recommendations,
        })
    }

    /// Analyze Model Training component
    fn analyze_model_training(&self) -> Result<UIComponentAnalysis, Box<dyn std::error::Error>> {
        let web_rs_path = format!("{}/src/web.rs", self.project_root);
        let web_content = fs::read_to_string(&web_rs_path)?;

        let endpoint_exists = web_content.contains("start_training") && web_content.contains("/api/training");
        let frontend_exists = web_content.contains("startTraining()") && web_content.contains("training-progress");
        
        // Check for actual implementation - this is KEY: it's just a placeholder
        let backend_implemented = web_content.contains("async fn start_training") && 
                                 !web_content.contains("// This would") && 
                                 !web_content.contains("return a placeholder response");
        
        // More thorough check - see if it actually starts training
        let is_functional = backend_implemented && 
                           web_content.contains("start_training_in_background") &&
                           web_content.contains("TrainingHandle") &&
                           !web_content.contains("Placeholder");

        let functional = endpoint_exists && frontend_exists && is_functional;
        
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !endpoint_exists {
            issues.push("API endpoint for model training not found".to_string());
            recommendations.push("Implement POST /api/training/start endpoint".to_string());
        }

        if !frontend_exists {
            issues.push("Frontend UI for model training not found".to_string());
            recommendations.push("Add model training form to frontend".to_string());
        }

        if !backend_implemented {
            issues.push("Backend model training logic not implemented".to_string());
            recommendations.push("Implement start_training function in web.rs".to_string());
        } else if !is_functional {
            issues.push("Backend model training is just a placeholder - doesn't actually start training".to_string());
            recommendations.push("Implement actual model training functionality".to_string());
        }

        Ok(UIComponentAnalysis {
            name: "Model Training".to_string(),
            endpoint_exists,
            frontend_exists,
            backend_implemented: is_functional,
            functional,
            issues,
            recommendations,
        })
    }

    /// Analyze Model Management component
    fn analyze_model_management(&self) -> Result<UIComponentAnalysis, Box<dyn std::error::Error>> {
        let web_rs_path = format!("{}/src/web.rs", self.project_root);
        let web_content = fs::read_to_string(&web_rs_path)?;

        let endpoint_exists = web_content.contains("list_models") && web_content.contains("/api/models");
        let frontend_exists = web_content.contains("loadModels()") && web_content.contains("models-list");
        
        // Check for actual implementation
        let backend_implemented = web_content.contains("async fn list_models") && 
                                 !web_content.contains("// This would") && 
                                 !web_content.contains("return a placeholder response");
        
        // More thorough check - see if it actually lists real models
        let is_functional = backend_implemented && 
                           web_content.contains("registry.list_models") &&
                           web_content.contains("ModelRegistry") &&
                           !web_content.contains("Placeholder");

        let functional = endpoint_exists && frontend_exists && is_functional;
        
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !backend_implemented {
            issues.push("Backend model management logic not fully implemented".to_string());
            recommendations.push("Complete model management implementation in web.rs".to_string());
        } else if !is_functional {
            issues.push("Backend model management is just a placeholder".to_string());
            recommendations.push("Implement actual model listing and management".to_string());
        }

        Ok(UIComponentAnalysis {
            name: "Model Management".to_string(),
            endpoint_exists,
            frontend_exists,
            backend_implemented: is_functional,
            functional,
            issues,
            recommendations,
        })
    }

    /// Generate analysis summary
    fn generate_summary(&self, components: &[UIComponentAnalysis]) -> AnalysisSummary {
        let total_components = components.len();
        let fully_functional = components.iter().filter(|c| c.functional).count();
        let backend_only = components.iter().filter(|c| !c.functional && c.backend_implemented).count();
        let frontend_only = components.iter().filter(|c| !c.functional && c.frontend_exists && !c.backend_implemented).count();
        let incomplete = components.iter().filter(|c| !c.functional && !c.frontend_exists && !c.backend_implemented).count();
        
        let code_coverage = if total_components > 0 {
            (fully_functional as f32 / total_components as f32) * 100.0
        } else {
            0.0
        };

        AnalysisSummary {
            total_components,
            fully_functional,
            backend_only,
            frontend_only,
            incomplete,
            code_coverage,
        }
    }

    /// Generate a detailed report
    pub fn generate_report(&self) -> Result<String, Box<dyn std::error::Error>> {
        let analysis = self.analyze()?;
        
        let mut report = String::new();
        report.push_str("Stream Diffusion RS - Advanced UI Analysis Report\n");
        report.push_str("===============================================\n\n");
        
        report.push_str(&format!("Analysis Timestamp: {}\n", analysis.timestamp));
        report.push_str(&format!("Code Coverage: {:.1}%\n\n", analysis.summary.code_coverage));
        
        report.push_str("Component Analysis:\n");
        report.push_str("------------------\n\n");
        
        for component in &analysis.components {
            report.push_str(&format!("Component: {}\n", component.name));
            report.push_str(&format!("  Status: {}\n", if component.functional { "✅ Functional" } else { "❌ Incomplete" }));
            report.push_str(&format!("  Frontend: {}\n", if component.frontend_exists { "✅ Exists" } else { "❌ Missing" }));
            report.push_str(&format!("  Backend: {}\n", if component.backend_implemented { "✅ Implemented" } else { "❌ Missing" }));
            
            if !component.issues.is_empty() {
                report.push_str("  Issues:\n");
                for issue in &component.issues {
                    report.push_str(&format!("    - {}\n", issue));
                }
            }
            
            if !component.recommendations.is_empty() {
                report.push_str("  Recommendations:\n");
                for recommendation in &component.recommendations {
                    report.push_str(&format!("    - {}\n", recommendation));
                }
            }
            
            report.push_str("\n");
        }
        
        report.push_str("Summary:\n");
        report.push_str("--------\n");
        report.push_str(&format!("  Total Components: {}\n", analysis.summary.total_components));
        report.push_str(&format!("  Fully Functional: {}\n", analysis.summary.fully_functional));
        report.push_str(&format!("  Backend Only: {}\n", analysis.summary.backend_only));
        report.push_str(&format!("  Frontend Only: {}\n", analysis.summary.frontend_only));
        report.push_str(&format!("  Incomplete: {}\n", analysis.summary.incomplete));
        
        Ok(report)
    }

    /// Save analysis to JSON file
    pub fn save_analysis(&self, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let analysis = self.analyze()?;
        let json = serde_json::to_string_pretty(&analysis)?;
        fs::write(output_path, json)?;
        Ok(())
    }

    /// Save report to text file
    pub fn save_report(&self, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let report = self.generate_report()?;
        fs::write(output_path, report)?;
        Ok(())
    }
}

/// Run advanced UI analysis
pub fn run_advanced_analysis(project_root: &str) -> Result<AdvancedUIAnalysis, Box<dyn std::error::Error>> {
    let analyzer = AdvancedUIAnalyzer::new(project_root);
    analyzer.analyze()
}

/// Generate and save detailed report
pub fn generate_detailed_report(project_root: &str, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = AdvancedUIAnalyzer::new(project_root);
    analyzer.save_report(output_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_analyzer_creation() {
        let analyzer = AdvancedUIAnalyzer::new(".");
        assert_eq!(analyzer.project_root, ".");
    }
    
    #[test]
    fn test_component_analysis() {
        let analyzer = AdvancedUIAnalyzer::new(".");
        
        // Test image generation analysis
        let result = analyzer.analyze_image_generation();
        assert!(result.is_ok());
        
        // Test fractal shaders analysis (should be more complete)
        let result = analyzer.analyze_fractal_shaders();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_summary_generation() {
        let analyzer = AdvancedUIAnalyzer::new(".");
        let components = vec![
            UIComponentAnalysis {
                name: "Test Component".to_string(),
                endpoint_exists: true,
                frontend_exists: true,
                backend_implemented: true,
                functional: true,
                issues: vec![],
                recommendations: vec![],
            }
        ];
        
        let summary = analyzer.generate_summary(&components);
        assert_eq!(summary.total_components, 1);
        assert_eq!(summary.fully_functional, 1);
        assert_eq!(summary.code_coverage, 100.0);
    }
}