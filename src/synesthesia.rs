//! Synesthetic Framework for Multimodal AI Integration
//! 
//! This module provides a comprehensive framework for connecting different sensory modalities
//! including gesture, vision, audio, EEG, and 3D models in a unified synesthetic experience.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// Synesthetic Framework
pub struct SynestheticFramework {
    /// Connection manager for different modalities
    connections: HashMap<SensoryModality, Box<dyn SensoryConnector>>,
    
    /// Event broadcaster for cross-modal communication
    event_sender: broadcast::Sender<SynestheticEvent>,
    
    /// Current state of all modalities
    state: Arc<Mutex<SynestheticState>>,
    
    /// Fusion strategies for combining sensory inputs
    fusion_strategies: HashMap<FusionType, Box<dyn FusionStrategy>>,
}

/// Sensory modalities supported by the framework
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensoryModality {
    Gesture,
    Vision,
    Audio,
    EEG,
    Model3D,
    Haptic,
    Thermal,
    Tactile,
}

/// Types of sensory fusion
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionType {
    Temporal,      // Time-based fusion
    Spatial,       // Space-based fusion
    Semantic,      // Meaning-based fusion
    Contextual,    // Context-based fusion
}

/// Synesthetic state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynestheticState {
    pub modalities_active: HashMap<SensoryModality, bool>,
    pub cross_modal_mappings: HashMap<SensoryModality, Vec<SensoryModality>>,
    pub last_events: HashMap<SensoryModality, SynestheticEvent>,
    pub fusion_weights: HashMap<FusionType, f32>,
}

/// Synesthetic event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynestheticEvent {
    pub modality: SensoryModality,
    pub data: EventData,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub confidence: f32,
    pub context: HashMap<String, String>,
}

/// Event data for different modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventData {
    /// Gesture event data
    Gesture {
        gesture_type: String,
        position: Option<(f32, f32, f32)>,
        velocity: Option<(f32, f32, f32)>,
        hand_id: Option<u32>,
    },
    
    /// Visual event data
    Visual {
        features: Vec<f32>,
        objects: Vec<VisualObject>,
        scene_description: String,
    },
    
    /// Audio event data
    Audio {
        features: Vec<f32>,
        frequency_bands: HashMap<String, f32>,
        rhythm: Option<RhythmPattern>,
    },
    
    /// EEG event data
    EEG {
        band_powers: HashMap<String, f32>,
        connectivity: Vec<f32>,
        emotional_state: Option<String>,
    },
    
    /// 3D model event data
    Model3D {
        model_id: String,
        transformation: (f32, f32, f32, f32), // Quaternion
        position: (f32, f32, f32),
        scale: (f32, f32, f32),
    },
    
    /// Haptic event data
    Haptic {
        intensity: f32,
        frequency: f32,
        duration: u32,
        pattern: String,
    },
}

/// Visual object detected in a scene
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualObject {
    pub id: String,
    pub class: String,
    pub bounding_box: (f32, f32, f32, f32), // x, y, width, height
    pub confidence: f32,
    pub features: Vec<f32>,
}

/// Rhythm pattern in audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmPattern {
    pub tempo: f32,
    pub beats: Vec<f32>,
    pub signature: String,
}

/// Trait for sensory connectors
pub trait SensoryConnector: Send + Sync {
    /// Connect to the sensory source
    fn connect(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Disconnect from the sensory source
    fn disconnect(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Process sensory input and generate events
    fn process_input(&self, input: &SensoryInput) -> Result<Vec<SynestheticEvent>, Box<dyn std::error::Error>>;
    
    /// Get connector information
    fn get_info(&self) -> SensoryConnectorInfo;
}

/// Information about a sensory connector
#[derive(Debug, Clone)]
pub struct SensoryConnectorInfo {
    pub modality: SensoryModality,
    pub name: String,
    pub version: String,
    pub requires_calibration: bool,
    pub supported_events: Vec<String>,
}

/// Sensory input data
#[derive(Debug)]
pub enum SensoryInput {
    /// Raw image data
    Image(ndarray::Array3<u8>),
    
    /// Audio samples
    Audio(Vec<f32>),
    
    /// EEG data
    EEG(crate::eeg::EEGData),
    
    /// Gesture data
    Gesture(crate::gesture::GestureInput),
    
    /// 3D model data
    Model3D(crate::ai_3d_models::Generated3DModel),
    
    /// Custom data
    Custom(Vec<u8>),
}

/// Trait for fusion strategies
pub trait FusionStrategy: Send + Sync {
    /// Fuse multiple sensory events
    fn fuse_events(&self, events: Vec<SynestheticEvent>) -> Result<SynestheticEvent, Box<dyn std::error::Error>>;
    
    /// Get strategy information
    fn get_info(&self) -> FusionStrategyInfo;
}

/// Information about a fusion strategy
#[derive(Debug, Clone)]
pub struct FusionStrategyInfo {
    pub fusion_type: FusionType,
    pub name: String,
    pub description: String,
    pub complexity: u32, // 1-10 scale
}

/// Cross-modal mapper
pub struct CrossModalMapper {
    /// Mappings between modalities
    mappings: HashMap<(SensoryModality, String), Vec<CrossModalMapping>>,
    
    /// Learning rate for adaptive mappings
    learning_rate: f32,
}

/// Cross-modal mapping
#[derive(Debug, Clone)]
pub struct CrossModalMapping {
    pub target_modality: SensoryModality,
    pub target_parameter: String,
    pub transformation: MappingTransformation,
    pub strength: f32, // 0.0 - 1.0
}

/// Mapping transformation types
#[derive(Debug, Clone)]
pub enum MappingTransformation {
    Linear(f32, f32),     // scale, offset
    Exponential(f32),     // exponent
    Threshold(f32),       // threshold value
    Custom(String),       // custom transformation function
}

impl SynestheticFramework {
    /// Create a new synesthetic framework
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (sender, _) = broadcast::channel(100);
        
        let state = Arc::new(Mutex::new(SynestheticState {
            modalities_active: HashMap::new(),
            cross_modal_mappings: HashMap::new(),
            last_events: HashMap::new(),
            fusion_weights: HashMap::new(),
        }));
        
        let mut framework = Self {
            connections: HashMap::new(),
            event_sender: sender,
            state,
            fusion_strategies: HashMap::new(),
        };
        
        // Initialize default fusion strategies
        framework.initialize_fusion_strategies()?;
        
        Ok(framework)
    }
    
    /// Initialize fusion strategies
    fn initialize_fusion_strategies(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.fusion_strategies.insert(
            FusionType::Temporal,
            Box::new(TemporalFusionStrategy::new()),
        );
        
        self.fusion_strategies.insert(
            FusionType::Spatial,
            Box::new(SpatialFusionStrategy::new()),
        );
        
        self.fusion_strategies.insert(
            FusionType::Semantic,
            Box::new(SemanticFusionStrategy::new()),
        );
        
        self.fusion_strategies.insert(
            FusionType::Contextual,
            Box::new(ContextualFusionStrategy::new()),
        );
        
        Ok(())
    }
    
    /// Add a sensory connector
    pub fn add_connector(&mut self, connector: Box<dyn SensoryConnector>) -> Result<(), Box<dyn std::error::Error>> {
        let info = connector.get_info();
        self.connections.insert(info.modality, connector);
        Ok(())
    }
    
    /// Activate a sensory modality
    pub fn activate_modality(&mut self, modality: SensoryModality) -> Result<(), Box<dyn std::error::Error>> {
        let modality_clone = modality.clone();
        if let Some(connector) = self.connections.get_mut(&modality_clone) {
            connector.connect()?;
            
            let mut state = self.state.lock().unwrap();
            state.modalities_active.insert(modality_clone.clone(), true);
            drop(state);
            
            log::info!("Activated {:?} modality", modality);
        }
        Ok(())
    }
    
    /// Deactivate a sensory modality
    pub fn deactivate_modality(&mut self, modality: SensoryModality) -> Result<(), Box<dyn std::error::Error>> {
        let modality_clone = modality.clone();
        if let Some(connector) = self.connections.get_mut(&modality_clone) {
            connector.disconnect()?;
            
            let mut state = self.state.lock().unwrap();
            state.modalities_active.insert(modality_clone.clone(), false);
            drop(state);
            
            log::info!("Deactivated {:?} modality", modality);
        }
        Ok(())
    }
    
    /// Process sensory input from a specific modality
    pub async fn process_input(&mut self, modality: SensoryModality, input: SensoryInput) -> Result<Vec<SynestheticEvent>, Box<dyn std::error::Error>> {
        let state = self.state.lock().unwrap();
        let is_active = *state.modalities_active.get(&modality).unwrap_or(&false);
        drop(state);
        
        if !is_active {
            return Ok(vec![]);
        }
        
        if let Some(connector) = self.connections.get(&modality) {
            let events = connector.process_input(&input)?;
            
            // Update state with latest events
            let mut state = self.state.lock().unwrap();
            for event in &events {
                state.last_events.insert(event.modality.clone(), event.clone());
            }
            drop(state);
            
            // Broadcast events
            for event in &events {
                let _ = self.event_sender.send(event.clone());
            }
            
            Ok(events)
        } else {
            Err(format!("No connector available for {:?} modality", modality).into())
        }
    }
    
    /// Get a subscription to synesthetic events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<SynestheticEvent> {
        self.event_sender.subscribe()
    }
    
    /// Apply cross-modal mappings
    pub async fn apply_cross_modal_mappings(&self, event: &SynestheticEvent) -> Result<Vec<SynestheticAction>, Box<dyn std::error::Error>> {
        let mut actions = Vec::new();
        
        match &event.data {
            // Gesture to other modalities mapping
            EventData::Gesture { gesture_type, position, velocity, hand_id: _ } => {
                match gesture_type.as_str() {
                    "swipe_left" => {
                        actions.push(SynestheticAction::Visual {
                            target: "diffusion".to_string(),
                            parameter: "hue".to_string(),
                            value: -0.1,
                        });
                    }
                    "swipe_right" => {
                        actions.push(SynestheticAction::Visual {
                            target: "diffusion".to_string(),
                            parameter: "hue".to_string(),
                            value: 0.1,
                        });
                    }
                    "swipe_up" => {
                        actions.push(SynestheticAction::Visual {
                            target: "diffusion".to_string(),
                            parameter: "brightness".to_string(),
                            value: 0.1,
                        });
                    }
                    "swipe_down" => {
                        actions.push(SynestheticAction::Visual {
                            target: "diffusion".to_string(),
                            parameter: "brightness".to_string(),
                            value: -0.1,
                        });
                    }
                    "pinch" => {
                        actions.push(SynestheticAction::Audio {
                            target: "synth".to_string(),
                            parameter: "frequency".to_string(),
                            value: 0.5,
                        });
                    }
                    "grab" => {
                        actions.push(SynestheticAction::Audio {
                            target: "synth".to_string(),
                            parameter: "amplitude".to_string(),
                            value: 0.8,
                        });
                    }
                    "victory" => {
                        actions.push(SynestheticAction::Model3D {
                            target: "current_model".to_string(),
                            parameter: "rotation".to_string(),
                            value: (0.1, 0.0, 0.0),
                        });
                    }
                    "thumbs_up" => {
                        actions.push(SynestheticAction::EEG {
                            target: "frontal".to_string(),
                            parameter: "alpha_power".to_string(),
                            value: 0.2,
                        });
                    }
                    "thumbs_down" => {
                        actions.push(SynestheticAction::EEG {
                            target: "frontal".to_string(),
                            parameter: "alpha_power".to_string(),
                            value: -0.2,
                        });
                    }
                    _ => {
                        // For unrecognized gestures, we can still map position/velocity to visual effects
                        if let Some((x, y, _)) = position {
                            actions.push(SynestheticAction::Visual {
                                target: "cursor".to_string(),
                                parameter: "position".to_string(),
                                value: (x + y) / 2.0, // Simple mapping
                            });
                        }
                    }
                }
            }
            
            // Visual to other modalities mapping
            EventData::Visual { features, objects: _, scene_description: _ } => {
                // Map visual features to audio parameters
                if !features.is_empty() {
                    let avg_feature = features.iter().sum::<f32>() / features.len() as f32;
                    actions.push(SynestheticAction::Audio {
                        target: "ambient".to_string(),
                        parameter: "filter_cutoff".to_string(),
                        value: avg_feature,
                    });
                }
            }
            
            // Audio to other modalities mapping
            EventData::Audio { features, frequency_bands, rhythm: _ } => {
                // Map audio features to visual parameters
                if !features.is_empty() {
                    let avg_feature = features.iter().sum::<f32>() / features.len() as f32;
                    actions.push(SynestheticAction::Visual {
                        target: "visualization".to_string(),
                        parameter: "intensity".to_string(),
                        value: avg_feature,
                    });
                }
                
                // Map frequency bands to 3D model parameters
                if let Some(bass) = frequency_bands.get("bass") {
                    actions.push(SynestheticAction::Model3D {
                        target: "bass_sphere".to_string(),
                        parameter: "scale".to_string(),
                        value: (*bass, *bass, *bass),
                    });
                }
            }
            
            // EEG to other modalities mapping
            EventData::EEG { band_powers, connectivity: _, emotional_state } => {
                // Map alpha waves to visual relaxation effects
                if let Some(alpha) = band_powers.get("alpha") {
                    actions.push(SynestheticAction::Visual {
                        target: "background".to_string(),
                        parameter: "saturation".to_string(),
                        value: *alpha,
                    });
                }
                
                // Map emotional state to music parameters
                if let Some(emotion) = emotional_state {
                    match emotion.as_str() {
                        "focused" => {
                            actions.push(SynestheticAction::Audio {
                                target: "focus_music".to_string(),
                                parameter: "tempo".to_string(),
                                value: 1.2, // Increase tempo
                            });
                        }
                        "relaxed" => {
                            actions.push(SynestheticAction::Audio {
                                target: "ambient".to_string(),
                                parameter: "reverb".to_string(),
                                value: 0.8, // Add reverb
                            });
                        }
                        _ => {}
                    }
                }
            }
            
            // 3D Model to other modalities mapping
            EventData::Model3D { model_id: _, transformation: _, position: _, scale: _ } => {
                // When a 3D model changes, we can trigger haptic feedback
                actions.push(SynestheticAction::Haptic {
                    intensity: 0.5,
                    duration: 100,
                    pattern: "pulse".to_string(),
                });
            }
            
            // Haptic to other modalities mapping
            EventData::Haptic { intensity, frequency: _, duration: _, pattern: _ } => {
                // Map haptic feedback to visual effects
                actions.push(SynestheticAction::Visual {
                    target: "feedback".to_string(),
                    parameter: "flash_intensity".to_string(),
                    value: *intensity,
                });
            }
        }
        
        Ok(actions)
    }
    
    /// Fuse events using a specific strategy
    pub async fn fuse_events(&self, events: Vec<SynestheticEvent>, fusion_type: FusionType) -> Result<SynestheticEvent, Box<dyn std::error::Error>> {
        if let Some(strategy) = self.fusion_strategies.get(&fusion_type) {
            strategy.fuse_events(events)
        } else {
            Err(format!("No fusion strategy available for {:?}", fusion_type).into())
        }
    }
}

/// Temporal fusion strategy
pub struct TemporalFusionStrategy {
    window_size: usize,
}

impl TemporalFusionStrategy {
    pub fn new() -> Self {
        Self {
            window_size: 10,
        }
    }
}

impl FusionStrategy for TemporalFusionStrategy {
    fn fuse_events(&self, events: Vec<SynestheticEvent>) -> Result<SynestheticEvent, Box<dyn std::error::Error>> {
        // In a real implementation, this would fuse events based on temporal proximity
        // For now, we'll return the most recent event
        if let Some(latest_event) = events.last() {
            Ok(latest_event.clone())
        } else {
            Err("No events to fuse".into())
        }
    }
    
    fn get_info(&self) -> FusionStrategyInfo {
        FusionStrategyInfo {
            fusion_type: FusionType::Temporal,
            name: "Temporal Fusion".to_string(),
            description: "Fuses events based on temporal proximity".to_string(),
            complexity: 3,
        }
    }
}

/// Spatial fusion strategy
pub struct SpatialFusionStrategy {
    proximity_threshold: f32,
}

impl SpatialFusionStrategy {
    pub fn new() -> Self {
        Self {
            proximity_threshold: 0.1,
        }
    }
}

impl FusionStrategy for SpatialFusionStrategy {
    fn fuse_events(&self, events: Vec<SynestheticEvent>) -> Result<SynestheticEvent, Box<dyn std::error::Error>> {
        // In a real implementation, this would fuse events based on spatial proximity
        // For now, we'll return the first event
        if let Some(first_event) = events.first() {
            Ok(first_event.clone())
        } else {
            Err("No events to fuse".into())
        }
    }
    
    fn get_info(&self) -> FusionStrategyInfo {
        FusionStrategyInfo {
            fusion_type: FusionType::Spatial,
            name: "Spatial Fusion".to_string(),
            description: "Fuses events based on spatial proximity".to_string(),
            complexity: 4,
        }
    }
}

/// Semantic fusion strategy
pub struct SemanticFusionStrategy {
    semantic_model: Option<String>, // In a real implementation, this would be a model
}

impl SemanticFusionStrategy {
    pub fn new() -> Self {
        Self {
            semantic_model: None,
        }
    }
}

impl FusionStrategy for SemanticFusionStrategy {
    fn fuse_events(&self, events: Vec<SynestheticEvent>) -> Result<SynestheticEvent, Box<dyn std::error::Error>> {
        // In a real implementation, this would fuse events based on semantic similarity
        // For now, we'll return a combined event
        if !events.is_empty() {
            Ok(SynestheticEvent {
                modality: SensoryModality::Gesture, // Placeholder
                data: EventData::Gesture {
                    gesture_type: "fused_gesture".to_string(),
                    position: None,
                    velocity: None,
                    hand_id: None,
                },
                timestamp: chrono::Utc::now(),
                confidence: events.iter().map(|e| e.confidence).sum::<f32>() / events.len() as f32,
                context: HashMap::new(),
            })
        } else {
            Err("No events to fuse".into())
        }
    }
    
    fn get_info(&self) -> FusionStrategyInfo {
        FusionStrategyInfo {
            fusion_type: FusionType::Semantic,
            name: "Semantic Fusion".to_string(),
            description: "Fuses events based on semantic similarity".to_string(),
            complexity: 8,
        }
    }
}

/// Contextual fusion strategy
pub struct ContextualFusionStrategy {
    context_model: Option<String>, // In a real implementation, this would be a model
}

impl ContextualFusionStrategy {
    pub fn new() -> Self {
        Self {
            context_model: None,
        }
    }
}

impl FusionStrategy for ContextualFusionStrategy {
    fn fuse_events(&self, events: Vec<SynestheticEvent>) -> Result<SynestheticEvent, Box<dyn std::error::Error>> {
        // In a real implementation, this would fuse events based on contextual relevance
        // For now, we'll return a context-aware event
        if !events.is_empty() {
            let mut context = HashMap::new();
            context.insert("fused_events".to_string(), events.len().to_string());
            
            Ok(SynestheticEvent {
                modality: SensoryModality::Gesture, // Placeholder
                data: EventData::Gesture {
                    gesture_type: "contextual_gesture".to_string(),
                    position: None,
                    velocity: None,
                    hand_id: None,
                },
                timestamp: chrono::Utc::now(),
                confidence: 0.9,
                context,
            })
        } else {
            Err("No events to fuse".into())
        }
    }
    
    fn get_info(&self) -> FusionStrategyInfo {
        FusionStrategyInfo {
            fusion_type: FusionType::Contextual,
            name: "Contextual Fusion".to_string(),
            description: "Fuses events based on contextual relevance".to_string(),
            complexity: 7,
        }
    }
}

/// Synesthetic action that can be executed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynestheticAction {
    /// Visual transformation
    Visual {
        target: String,
        parameter: String,
        value: f32,
    },
    
    /// Audio transformation
    Audio {
        target: String,
        parameter: String,
        value: f32,
    },
    
    /// 3D model transformation
    Model3D {
        target: String,
        parameter: String,
        value: (f32, f32, f32),
    },
    
    /// EEG modulation
    EEG {
        target: String,
        parameter: String,
        value: f32,
    },
    
    /// Haptic feedback
    Haptic {
        intensity: f32,
        duration: u32,
        pattern: String,
    },
}

/// Synesthetic experience controller
pub struct SynestheticController {
    framework: SynestheticFramework,
    mapper: CrossModalMapper,
    actions: Vec<SynestheticAction>,
}

impl SynestheticController {
    /// Create a new synesthetic controller
    pub fn new(framework: SynestheticFramework) -> Result<Self, Box<dyn std::error::Error>> {
        let mapper = CrossModalMapper {
            mappings: HashMap::new(),
            learning_rate: 0.01,
        };
        
        Ok(Self {
            framework,
            mapper,
            actions: Vec::new(),
        })
    }
    
    /// Process synesthetic events and generate actions
    pub async fn process_events(&mut self, events: Vec<SynestheticEvent>) -> Result<Vec<SynestheticAction>, Box<dyn std::error::Error>> {
        let mut actions = Vec::new();
        
        for event in events {
            // Apply cross-modal mappings
            let mapped_actions = self.framework.apply_cross_modal_mappings(&event).await?;
            actions.extend(mapped_actions);
        }
        
        // Store actions for execution
        self.actions.extend(actions.clone());
        
        Ok(actions)
    }
    
    /// Execute synesthetic actions
    pub async fn execute_actions(&self) -> Result<(), Box<dyn std::error::Error>> {
        for action in &self.actions {
            match action {
                SynestheticAction::Visual { target, parameter, value } => {
                    log::info!("Executing visual action: {}::{} = {}", target, parameter, value);
                    // In a real implementation, this would control visual components
                }
                SynestheticAction::Audio { target, parameter, value } => {
                    log::info!("Executing audio action: {}::{} = {}", target, parameter, value);
                    // In a real implementation, this would control audio components
                }
                SynestheticAction::Model3D { target, parameter, value } => {
                    log::info!("Executing 3D model action: {}::{} = {:?}", target, parameter, value);
                    // In a real implementation, this would control 3D models
                }
                SynestheticAction::EEG { target, parameter, value } => {
                    log::info!("Executing EEG action: {}::{} = {}", target, parameter, value);
                    // In a real implementation, this would control EEG parameters
                }
                SynestheticAction::Haptic { intensity, duration, pattern } => {
                    log::info!("Executing haptic action: {} {} for {}ms", pattern, intensity, duration);
                    // In a real implementation, this would control haptic devices
                }
            }
        }
        
        Ok(())
    }
    
    /// Clear executed actions
    pub fn clear_actions(&mut self) {
        self.actions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_synesthetic_framework_creation() {
        let framework = SynestheticFramework::new();
        assert!(framework.is_ok());
    }
    
    #[test]
    fn test_fusion_strategies_initialization() {
        let mut framework = SynestheticFramework::new().unwrap();
        assert!(framework.initialize_fusion_strategies().is_ok());
        assert!(framework.fusion_strategies.contains_key(&FusionType::Temporal));
        assert!(framework.fusion_strategies.contains_key(&FusionType::Spatial));
        assert!(framework.fusion_strategies.contains_key(&FusionType::Semantic));
        assert!(framework.fusion_strategies.contains_key(&FusionType::Contextual));
    }
    
    #[test]
    fn test_temporal_fusion_strategy() {
        let strategy = TemporalFusionStrategy::new();
        let info = strategy.get_info();
        assert_eq!(info.fusion_type, FusionType::Temporal);
        assert_eq!(info.name, "Temporal Fusion");
    }
    
    #[test]
    fn test_spatial_fusion_strategy() {
        let strategy = SpatialFusionStrategy::new();
        let info = strategy.get_info();
        assert_eq!(info.fusion_type, FusionType::Spatial);
        assert_eq!(info.name, "Spatial Fusion");
    }
    
    #[test]
    fn test_semantic_fusion_strategy() {
        let strategy = SemanticFusionStrategy::new();
        let info = strategy.get_info();
        assert_eq!(info.fusion_type, FusionType::Semantic);
        assert_eq!(info.name, "Semantic Fusion");
    }
    
    #[test]
    fn test_contextual_fusion_strategy() {
        let strategy = ContextualFusionStrategy::new();
        let info = strategy.get_info();
        assert_eq!(info.fusion_type, FusionType::Contextual);
        assert_eq!(info.name, "Contextual Fusion");
    }
    
    #[test]
    fn test_synesthetic_event_creation() {
        let event = SynestheticEvent {
            modality: SensoryModality::Gesture,
            data: EventData::Gesture {
                gesture_type: "swipe_left".to_string(),
                position: Some((0.5, 0.5, 0.0)),
                velocity: Some((0.1, 0.0, 0.0)),
                hand_id: Some(1),
            },
            timestamp: chrono::Utc::now(),
            confidence: 0.95,
            context: HashMap::new(),
        };
        
        match event.data {
            EventData::Gesture { gesture_type, position, velocity, hand_id } => {
                assert_eq!(gesture_type, "swipe_left");
                assert_eq!(position, Some((0.5, 0.5, 0.0)));
                assert_eq!(velocity, Some((0.1, 0.0, 0.0)));
                assert_eq!(hand_id, Some(1));
            }
            _ => panic!("Wrong event data type"),
        }
    }
}