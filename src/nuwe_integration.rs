//! NUWE Integration Module for Stream Diffusion RS
//!
//! This module provides integration with NUWE's node-based architecture,
//! allowing Stream Diffusion's EEG processing and audiovisual conversion
//! capabilities to be used as nodes in NUWE pipelines.

use crate::eeg::{EEGData, EEGProcessor, EEGToAudiovisualConverter, RealtimeEEGProcessor, EEGVisualizer};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// NUWE-compatible node types for EEG and audiovisual processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NuweNodeType {
    EEGProcessor,
    EEGVisualizer,
    AudiovisualConverter,
    RealtimeEEGProcessor,
    EEGFeatureExtractor,
}

/// NUWE node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NuweNodeConfig {
    pub node_type: NuweNodeType,
    pub parameters: HashMap<String, Value>,
}

/// NUWE node instance that wraps Stream Diffusion functionality
pub struct NuweEEGNode {
    pub id: String,
    pub node_type: NuweNodeType,
    pub config: NuweNodeConfig,

    // Internal processors
    eeg_processor: Option<EEGProcessor>,
    visualizer: Option<EEGVisualizer>,
    converter: Option<EEGToAudiovisualConverter>,
    realtime_processor: Option<RealtimeEEGProcessor>,

    // Shared state
    current_eeg_data: Arc<Mutex<Option<EEGData>>>,
    output_dir: std::path::PathBuf,
}

impl NuweEEGNode {
    /// Create a new NUWE EEG node
    pub fn new(id: String, config: NuweNodeConfig, output_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut node = Self {
            id,
            node_type: config.node_type.clone(),
            config,
            eeg_processor: None,
            visualizer: None,
            converter: None,
            realtime_processor: None,
            current_eeg_data: Arc::new(Mutex::new(None)),
            output_dir: output_dir.to_path_buf(),
        };

        node.initialize_components()?;
        Ok(node)
    }

    /// Initialize the internal components based on node type
    fn initialize_components(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self.node_type {
            NuweNodeType::EEGProcessor => {
                self.eeg_processor = Some(EEGProcessor::new());
            }
            NuweNodeType::EEGVisualizer => {
                self.visualizer = Some(EEGVisualizer::new(&self.output_dir));
            }
            NuweNodeType::AudiovisualConverter => {
                self.converter = Some(EEGToAudiovisualConverter::new(&self.output_dir)?);
            }
            NuweNodeType::RealtimeEEGProcessor => {
                let buffer_size = self.config.parameters
                    .get("buffer_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1000) as usize;
                self.realtime_processor = Some(RealtimeEEGProcessor::new(buffer_size));
            }
            NuweNodeType::EEGFeatureExtractor => {
                // Feature extractor combines processor and converter
                self.eeg_processor = Some(EEGProcessor::new());
                self.converter = Some(EEGToAudiovisualConverter::new(&self.output_dir)?);
            }
        }
        Ok(())
    }

    /// Process input data (NUWE-compatible interface)
    pub fn process(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
        match self.node_type {
            NuweNodeType::EEGProcessor => self.process_eeg_data(inputs),
            NuweNodeType::EEGVisualizer => self.process_visualization(inputs),
            NuweNodeType::AudiovisualConverter => self.process_audiovisual_conversion(inputs),
            NuweNodeType::RealtimeEEGProcessor => self.process_realtime_eeg(inputs),
            NuweNodeType::EEGFeatureExtractor => self.process_feature_extraction(inputs),
        }
    }

    /// Process EEG data through the processor
    fn process_eeg_data(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
        let processor = self.eeg_processor.as_mut().ok_or("EEG processor not initialized")?;

        // Extract EEG data from inputs
    if let Some(eeg_data_value) = inputs.get("eeg_data") {
        // For now, create dummy EEG data - in practice would parse from JSON
        let eeg_data = EEGData::new(
            ndarray::Array3::zeros((32, 1000, 1)),
            250.0,
            (0..32).map(|i| format!("Ch{}", i + 1)).collect(),
        );

            // Apply processing
            let mut processed_data = eeg_data.clone();
            processor.remove_dc_offset(&mut processed_data);

            // Store for other nodes
            *self.current_eeg_data.lock().unwrap() = Some(processed_data.clone());

            // Return processed data
            let output = serde_json::to_value(processed_data)?;
            Ok(HashMap::from([("processed_eeg".to_string(), output)]))
        } else {
            Err("No EEG data provided".into())
        }
    }

    /// Generate visualizations
    fn process_visualization(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
        let visualizer = self.visualizer.as_ref().ok_or("Visualizer not initialized")?;

        if let Some(eeg_data_value) = inputs.get("eeg_data") {
            let eeg_data: EEGData = serde_json::from_value(eeg_data_value.clone())?;

            // Generate visualizations
            let output_files = vec![
                format!("{}_topography.png", self.id),
                format!("{}_psd.png", self.id),
            ];

            // Plot topography
            visualizer.plot_topography(&eeg_data, 0, &output_files[0])?;

            // Plot PSD for first channel
            let (freqs, psd) = self.eeg_processor.as_ref()
                .ok_or("EEG processor needed for PSD")?
                .compute_psd(&eeg_data, 0, 0)?;
            visualizer.plot_psd(&freqs, &psd, "EEG001", &output_files[1])?;

            Ok(HashMap::from([("visualization_files".to_string(), serde_json::to_value(output_files)?)]))
        } else {
            Err("No EEG data provided".into())
        }
    }

    /// Convert EEG to audiovisual output
    fn process_audiovisual_conversion(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
        let converter = self.converter.as_ref().ok_or("Converter not initialized")?;

        if let Some(_eeg_data_value) = inputs.get("eeg_data") {
            let eeg_data = EEGData::new(
                ndarray::Array3::zeros((32, 1000, 1)),
                250.0,
                (0..32).map(|i| format!("Ch{}", i + 1)).collect(),
            );

            // Convert to audiovisual
            let audiovisual_data = converter.convert(&eeg_data)?;

            // Return audiovisual data
            let output = HashMap::from([
                ("visual_data".to_string(), serde_json::to_value(audiovisual_data.visual)?),
                ("audio_data".to_string(), serde_json::to_value(audiovisual_data.audio)?),
                ("features".to_string(), serde_json::to_value(audiovisual_data.features)?),
            ]);

            Ok(output)
        } else {
            Err("No EEG data provided".into())
        }
    }

    /// Process real-time EEG data
    fn process_realtime_eeg(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
        let processor = self.realtime_processor.as_mut().ok_or("Realtime processor not initialized")?;

        if let Some(sample_value) = inputs.get("eeg_sample") {
            let sample: f32 = serde_json::from_value(sample_value.clone())?;
            processor.add_sample(sample);

            // Process buffer if ready
            let features = processor.process_buffer()?;

            Ok(HashMap::from([
                ("realtime_features".to_string(), serde_json::to_value(features)?),
                ("buffer_size".to_string(), serde_json::to_value(processor.get_buffer().len())?),
            ]))
        } else {
            Err("No EEG sample provided".into())
        }
    }

    /// Extract features from EEG data
    fn process_feature_extraction(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
        let processor = self.eeg_processor.as_ref().ok_or("EEG processor not initialized")?;
        let converter = self.converter.as_ref().ok_or("Converter not initialized")?;

        if let Some(_eeg_data_value) = inputs.get("eeg_data") {
            let eeg_data = EEGData::new(
                ndarray::Array3::zeros((32, 1000, 1)),
                250.0,
                (0..32).map(|i| format!("Ch{}", i + 1)).collect(),
            );

            // Extract features (simplified - would use actual method)
            let features = crate::eeg::EEGFeatures {
                band_powers: std::collections::HashMap::new(),
                connectivity: vec![0.0; 32 * 32],
                complexity: vec![0.0; 32],
            };

            // Compute additional statistics
            let statistics = processor.compute_statistics(&eeg_data);

            let output = HashMap::from([
                ("band_powers".to_string(), serde_json::to_value(&features.band_powers)?),
                ("connectivity".to_string(), serde_json::to_value(&features.connectivity)?),
                ("complexity".to_string(), serde_json::to_value(&features.complexity)?),
                ("statistics".to_string(), Value::String("eeg_statistics".to_string())),
            ]);

            Ok(output)
        } else {
            Err("No EEG data provided".into())
        }
    }

    /// Get node metadata for NUWE
    pub fn get_node_info(&self) -> NuweNodeInfo {
        let (name, description, inputs, outputs) = match self.node_type {
            NuweNodeType::EEGProcessor => (
                "EEG Processor".to_string(),
                "Processes EEG signals with filtering and artifact removal".to_string(),
                vec!["eeg_data".to_string()],
                vec!["processed_eeg".to_string()],
            ),
            NuweNodeType::EEGVisualizer => (
                "EEG Visualizer".to_string(),
                "Creates visualizations of EEG data".to_string(),
                vec!["eeg_data".to_string()],
                vec!["visualization_files".to_string()],
            ),
            NuweNodeType::AudiovisualConverter => (
                "EEG to Audiovisual".to_string(),
                "Converts EEG signals to synchronized audio and visual output".to_string(),
                vec!["eeg_data".to_string()],
                vec!["visual_data".to_string(), "audio_data".to_string(), "features".to_string()],
            ),
            NuweNodeType::RealtimeEEGProcessor => (
                "Realtime EEG Processor".to_string(),
                "Processes streaming EEG data in real-time".to_string(),
                vec!["eeg_sample".to_string()],
                vec!["realtime_features".to_string(), "buffer_size".to_string()],
            ),
            NuweNodeType::EEGFeatureExtractor => (
                "EEG Feature Extractor".to_string(),
                "Extracts frequency bands, connectivity, and complexity features".to_string(),
                vec!["eeg_data".to_string()],
                vec!["band_powers".to_string(), "connectivity".to_string(), "complexity".to_string(), "statistics".to_string()],
            ),
        };

        NuweNodeInfo {
            id: self.id.clone(),
            name,
            description,
            inputs,
            outputs,
            node_type: self.node_type.clone(),
        }
    }
}

/// Node information for NUWE interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NuweNodeInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub node_type: NuweNodeType,
}

/// Factory for creating NUWE nodes
pub struct NuweNodeFactory {
    output_dir: std::path::PathBuf,
}

impl NuweNodeFactory {
    pub fn new(output_dir: &Path) -> Self {
        Self {
            output_dir: output_dir.to_path_buf(),
        }
    }

    pub fn create_node(&self, id: String, config: NuweNodeConfig) -> Result<NuweEEGNode, Box<dyn std::error::Error>> {
        NuweEEGNode::new(id, config, &self.output_dir)
    }

    pub fn get_available_node_types(&self) -> Vec<NuweNodeType> {
        vec![
            NuweNodeType::EEGProcessor,
            NuweNodeType::EEGVisualizer,
            NuweNodeType::AudiovisualConverter,
            NuweNodeType::RealtimeEEGProcessor,
            NuweNodeType::EEGFeatureExtractor,
        ]
    }
}

/// Integration manager for NUWE in Stream Diffusion
pub struct NuweIntegrationManager {
    factory: NuweNodeFactory,
    active_nodes: HashMap<String, NuweEEGNode>,
}

impl NuweIntegrationManager {
    pub fn new(output_dir: &Path) -> Self {
        Self {
            factory: NuweNodeFactory::new(output_dir),
            active_nodes: HashMap::new(),
        }
    }

    pub fn create_node(&mut self, id: String, config: NuweNodeConfig) -> Result<(), Box<dyn std::error::Error>> {
        let node = self.factory.create_node(id.clone(), config)?;
        self.active_nodes.insert(id, node);
        Ok(())
    }

    pub fn process_node(&mut self, node_id: &str, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
        let node = self.active_nodes.get_mut(node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;
        node.process(inputs)
    }

    pub fn get_node_info(&self, node_id: &str) -> Option<NuweNodeInfo> {
        self.active_nodes.get(node_id).map(|node| node.get_node_info())
    }

    pub fn list_active_nodes(&self) -> Vec<String> {
        self.active_nodes.keys().cloned().collect()
    }

    pub fn remove_node(&mut self, node_id: &str) -> bool {
        self.active_nodes.remove(node_id).is_some()
    }
}