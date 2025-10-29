//! Bevy Engine Integration for Stream Diffusion RS
//!
//! This module provides integration with the Bevy game engine for 3D rendering,
//! physics simulation, and real-time 3D AI synthesis. Adapted from NUWE's Bevy implementation.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Bevy Scene Manager for 3D rendering and physics
pub struct BevySceneManager {
    scenes: HashMap<String, BevyScene>,
    active_scene: Option<String>,
    physics_enabled: bool,
    rendering_enabled: bool,
}

impl BevySceneManager {
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
            active_scene: None,
            physics_enabled: true,
            rendering_enabled: true,
        }
    }

    /// Create a new 3D scene
    pub fn create_scene(&mut self, scene_id: &str, config: SceneConfig) -> Result<(), Box<dyn std::error::Error>> {
        let scene = BevyScene::new(scene_id.to_string(), config);
        self.scenes.insert(scene_id.to_string(), scene);
        Ok(())
    }

    /// Load scene by ID
    pub fn load_scene(&mut self, scene_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        if self.scenes.contains_key(scene_id) {
            self.active_scene = Some(scene_id.to_string());
            Ok(())
        } else {
            Err(format!("Scene {} not found", scene_id).into())
        }
    }

    /// Add entity to active scene
    pub fn add_entity(&mut self, entity: BevyEntity) -> Result<String, Box<dyn std::error::Error>> {
        if let Some(scene_id) = &self.active_scene {
            if let Some(scene) = self.scenes.get_mut(scene_id) {
                let entity_id = format!("entity_{}", scene.entities.len());
                scene.entities.insert(entity_id.clone(), entity);
                Ok(entity_id)
            } else {
                Err("Active scene not found".into())
            }
        } else {
            Err("No active scene".into())
        }
    }

    /// Update entity in active scene
    pub fn update_entity(&mut self, entity_id: &str, transform: Transform, properties: HashMap<String, Value>) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(scene_id) = &self.active_scene {
            if let Some(scene) = self.scenes.get_mut(scene_id) {
                if let Some(entity) = scene.entities.get_mut(entity_id) {
                    entity.transform = transform;
                    entity.properties.extend(properties);
                    Ok(())
                } else {
                    Err(format!("Entity {} not found", entity_id).into())
                }
            } else {
                Err("Active scene not found".into())
            }
        } else {
            Err("No active scene".into())
        }
    }

    /// Apply physics to active scene
    pub fn update_physics(&mut self, delta_time: f32) -> Result<(), Box<dyn std::error::Error>> {
        if !self.physics_enabled {
            return Ok(());
        }

        if let Some(scene_id) = &self.active_scene {
            if let Some(scene) = self.scenes.get_mut(scene_id) {
                for entity in scene.entities.values_mut() {
                    if let Some(velocity) = entity.properties.get("velocity") {
                        if let Some(vel_array) = velocity.as_array() {
                            if vel_array.len() >= 3 {
                                let vx = vel_array[0].as_f64().unwrap_or(0.0) as f32;
                                let vy = vel_array[1].as_f64().unwrap_or(0.0) as f32;
                                let vz = vel_array[2].as_f64().unwrap_or(0.0) as f32;

                                entity.transform.position.0 += vx * delta_time;
                                entity.transform.position.1 += vy * delta_time;
                                entity.transform.position.2 += vz * delta_time;
                            }
                        }
                    }
                }
                Ok(())
            } else {
                Err("Active scene not found".into())
            }
        } else {
            Err("No active scene".into())
        }
    }

    /// Get scene data for rendering
    pub fn get_scene_data(&self) -> Result<Option<SceneData>, Box<dyn std::error::Error>> {
        if let Some(scene_id) = &self.active_scene {
            if let Some(scene) = self.scenes.get(scene_id) {
                let entities: Vec<EntityData> = scene.entities.values().map(|e| EntityData {
                    id: e.id.clone(),
                    transform: e.transform.clone(),
                    mesh_type: e.mesh_type.clone(),
                    material: e.material.clone(),
                    properties: e.properties.clone(),
                }).collect();

                Ok(Some(SceneData {
                    scene_id: scene.id.clone(),
                    entities,
                    camera: scene.camera.clone(),
                    lighting: scene.lighting.clone(),
                }))
            } else {
                Err("Active scene not found".into())
            }
        } else {
            Ok(None)
        }
    }

    /// Enable/disable physics
    pub fn set_physics_enabled(&mut self, enabled: bool) {
        self.physics_enabled = enabled;
    }

    /// Enable/disable rendering
    pub fn set_rendering_enabled(&mut self, enabled: bool) {
        self.rendering_enabled = enabled;
    }
}

/// Bevy Scene representation
pub struct BevyScene {
    pub id: String,
    pub config: SceneConfig,
    pub entities: HashMap<String, BevyEntity>,
    pub camera: Camera,
    pub lighting: Lighting,
}

impl BevyScene {
    pub fn new(id: String, config: SceneConfig) -> Self {
        Self {
            id,
            config,
            entities: HashMap::new(),
            camera: Camera::default(),
            lighting: Lighting::default(),
        }
    }
}

/// Scene Configuration
#[derive(Debug, Clone)]
pub struct SceneConfig {
    pub name: String,
    pub gravity: (f32, f32, f32),
    pub ambient_light: (f32, f32, f32, f32),
    pub clear_color: (f32, f32, f32, f32),
}

/// Bevy Entity representation
pub struct BevyEntity {
    pub id: String,
    pub transform: Transform,
    pub mesh_type: MeshType,
    pub material: Material,
    pub properties: HashMap<String, Value>,
}

impl BevyEntity {
    pub fn new(id: String, mesh_type: MeshType) -> Self {
        Self {
            id,
            transform: Transform::default(),
            mesh_type,
            material: Material::default(),
            properties: HashMap::new(),
        }
    }
}

/// 3D Transform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    pub position: (f32, f32, f32),
    pub rotation: (f32, f32, f32, f32), // quaternion
    pub scale: (f32, f32, f32),
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: (0.0, 0.0, 0.0),
            rotation: (0.0, 0.0, 0.0, 1.0),
            scale: (1.0, 1.0, 1.0),
        }
    }
}

/// Mesh types for 3D rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshType {
    Cube,
    Sphere,
    Plane,
    Custom(String),
}

/// Material properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    pub color: (f32, f32, f32, f32),
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: (f32, f32, f32),
}

impl Default for Material {
    fn default() -> Self {
        Self {
            color: (1.0, 1.0, 1.0, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            emissive: (0.0, 0.0, 0.0),
        }
    }
}

/// Camera configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera {
    pub position: (f32, f32, f32),
    pub target: (f32, f32, f32),
    pub up: (f32, f32, f32),
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: (0.0, 0.0, 5.0),
            target: (0.0, 0.0, 0.0),
            up: (0.0, 1.0, 0.0),
            fov: 60.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

/// Lighting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lighting {
    pub directional_light: Option<DirectionalLight>,
    pub point_lights: Vec<PointLight>,
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            directional_light: Some(DirectionalLight::default()),
            point_lights: vec![],
        }
    }
}

/// Directional light
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalLight {
    pub direction: (f32, f32, f32),
    pub color: (f32, f32, f32),
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: (-1.0, -1.0, -1.0),
            color: (1.0, 1.0, 1.0),
            intensity: 1.0,
        }
    }
}

/// Point light
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointLight {
    pub position: (f32, f32, f32),
    pub color: (f32, f32, f32),
    pub intensity: f32,
    pub range: f32,
}

/// Scene data for frontend rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneData {
    pub scene_id: String,
    pub entities: Vec<EntityData>,
    pub camera: Camera,
    pub lighting: Lighting,
}

/// Entity data for frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityData {
    pub id: String,
    pub transform: Transform,
    pub mesh_type: MeshType,
    pub material: Material,
    pub properties: HashMap<String, Value>,
}

/// EEG Control for Bevy scenes
pub struct BevyEEGController {
    scene_manager: BevySceneManager,
    eeg_mapping: HashMap<String, EEGMapping>,
    control_enabled: bool,
}

impl BevyEEGController {
    pub fn new(scene_manager: BevySceneManager) -> Self {
        Self {
            scene_manager,
            eeg_mapping: HashMap::new(),
            control_enabled: false,
        }
    }

    /// Enable EEG control
    pub fn enable_control(&mut self) {
        self.control_enabled = true;
    }

    /// Disable EEG control
    pub fn disable_control(&mut self) {
        self.control_enabled = false;
    }

    /// Calibrate EEG mappings
    pub fn calibrate_mapping(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Reset mappings to baseline
        self.eeg_mapping.clear();

        // Create default mappings
        self.eeg_mapping.insert("alpha_power".to_string(), EEGMapping {
            parameter: "camera_position_z".to_string(),
            min_value: 0.0,
            max_value: 50.0,
            sensitivity: 2.0,
        });

        self.eeg_mapping.insert("beta_power".to_string(), EEGMapping {
            parameter: "light_intensity".to_string(),
            min_value: 0.0,
            max_value: 2.0,
            sensitivity: 1.5,
        });

        Ok(())
    }

    /// Process EEG data and update scene
    pub fn process_eeg_data(&mut self, eeg_data: &HashMap<String, f32>) -> Result<(), Box<dyn std::error::Error>> {
        if !self.control_enabled {
            return Ok(());
        }

        let mappings: Vec<(String, EEGMapping)> = self.eeg_mapping.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        for (eeg_param, mapping) in mappings {
            if let Some(value) = eeg_data.get(&eeg_param) {
                self.apply_eeg_mapping(&mapping, *value)?;
            }
        }

        Ok(())
    }

    fn apply_eeg_mapping(&mut self, mapping: &EEGMapping, value: f32) -> Result<(), Box<dyn std::error::Error>> {
        let normalized_value = (value - mapping.min_value) / (mapping.max_value - mapping.min_value);
        let scaled_value = normalized_value * mapping.sensitivity;

        match mapping.parameter.as_str() {
            "camera_position_z" => {
                // Update camera Z position based on alpha power
                if let Some(scene_id) = &self.scene_manager.active_scene {
                    if let Some(scene) = self.scene_manager.scenes.get_mut(scene_id) {
                        scene.camera.position.2 = 5.0 - scaled_value.clamp(-2.0, 2.0);
                    }
                }
            }
            "light_intensity" => {
                // Update light intensity based on beta power
                if let Some(scene_id) = &self.scene_manager.active_scene {
                    if let Some(scene) = self.scene_manager.scenes.get_mut(scene_id) {
                        if let Some(light) = &mut scene.lighting.directional_light {
                            light.intensity = scaled_value.clamp(0.1, 2.0);
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }
}

/// EEG to Bevy parameter mapping
#[derive(Debug, Clone)]
pub struct EEGMapping {
    pub parameter: String,
    pub min_value: f32,
    pub max_value: f32,
    pub sensitivity: f32,
}

/// Glicol Audio Integration
pub struct GlicolAudioEngine {
    audio_graph: HashMap<String, AudioNode>,
    sample_rate: u32,
    is_playing: bool,
}

impl GlicolAudioEngine {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            audio_graph: HashMap::new(),
            sample_rate,
            is_playing: false,
        }
    }

    /// Add audio node to graph
    pub fn add_node(&mut self, id: &str, node: AudioNode) -> Result<(), Box<dyn std::error::Error>> {
        self.audio_graph.insert(id.to_string(), node);
        Ok(())
    }

    /// Connect audio nodes
    pub fn connect_nodes(&mut self, from: &str, to: &str, output: usize, input: usize) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(from_node) = self.audio_graph.get_mut(from) {
            from_node.outputs.push((to.to_string(), output, input));
        }
        if let Some(to_node) = self.audio_graph.get_mut(to) {
            to_node.inputs.push((from.to_string(), input, output));
        }
        Ok(())
    }

    /// Start audio playback
    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.is_playing = true;
        Ok(())
    }

    /// Stop audio playback
    pub fn stop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.is_playing = false;
        Ok(())
    }

    /// Process audio for one frame
    pub fn process_frame(&mut self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if !self.is_playing {
            return Ok(vec![0.0; 128]); // Silent frame
        }

        // Simple audio processing - in real implementation would use Glicol
        let mut output = vec![0.0; 128];
        for node in self.audio_graph.values_mut() {
            if let Some(audio) = node.process()? {
                for (i, sample) in audio.iter().enumerate() {
                    if i < output.len() {
                        output[i] += sample;
                    }
                }
            }
        }

        Ok(output)
    }
}

/// Audio node for Glicol integration
pub struct AudioNode {
    pub node_type: String,
    pub parameters: HashMap<String, f32>,
    pub inputs: Vec<(String, usize, usize)>, // (from_node, input_index, output_index)
    pub outputs: Vec<(String, usize, usize)>, // (to_node, output_index, input_index)
}

impl AudioNode {
    pub fn new(node_type: &str) -> Self {
        Self {
            node_type: node_type.to_string(),
            parameters: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Process audio for this node
    pub fn process(&mut self) -> Result<Option<Vec<f32>>, Box<dyn std::error::Error>> {
        match self.node_type.as_str() {
            "sin" => {
                let freq = self.parameters.get("freq").unwrap_or(&440.0);
                let amp = self.parameters.get("amp").unwrap_or(&0.5);
                let mut output = vec![0.0; 128];

                for i in 0..128 {
                    let phase = (i as f32 / 44100.0) * freq * 2.0 * std::f32::consts::PI;
                    output[i] = phase.sin() * amp;
                }

                Ok(Some(output))
            }
            "noise" => {
                let amp = self.parameters.get("amp").unwrap_or(&0.3);
                let mut output = vec![0.0; 128];

                for i in 0..128 {
                    output[i] = (rand::random::<f32>() * 2.0 - 1.0) * amp;
                }

                Ok(Some(output))
            }
            _ => Ok(None),
        }
    }
}