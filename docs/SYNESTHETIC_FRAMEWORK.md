## 🎯 Usage Examples

### Basic Setup

```rust
use stream_diffusion_rs::synesthesia::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the synesthetic framework
    let mut framework = SynestheticFramework::new()?;
    
    // Activate modalities
    framework.activate_modality(SensoryModality::Gesture)?;
    framework.activate_modality(SensoryModality::Audio)?;
    framework.activate_modality(SensoryModality::EEG)?;
    
    // Process gesture input
    let gesture_input = SensoryInput::Gesture(GestureInput {
        hand_positions: vec![(100.0, 200.0, 0.0)],
        gesture_type: "swipe_left".to_string(),
    });
    
    let events = framework.process_input(SensoryModality::Gesture, gesture_input).await?;
    
    // Apply cross-modal mappings
    for event in events {
        let actions = framework.apply_cross_modal_mappings(&event).await?;
        println!("Generated {} actions", actions.len());
    }
    
    Ok(())
}
```

### Fusion Processing

```rust
use stream_diffusion_rs::synesthesia::*;

async fn process_multimodal_data(
    framework: &SynestheticFramework,
    gesture_events: Vec<SynestheticEvent>,
    audio_events: Vec<SynestheticEvent>,
) -> Result<SynestheticEvent, Box<dyn std::error::Error>> {
    // Combine events from different modalities
    let mut all_events = gesture_events;
    all_events.extend(audio_events);
    
    // Fuse events using temporal strategy
    let fused_event = framework.fuse_events(
        all_events, 
        FusionType::Temporal
    ).await?;
    
    Ok(fused_event)
}
```

### Cross-Modal Mapping

```rust
use stream_diffusion_rs::synesthesia::*;

async fn handle_gesture_event(
    framework: &SynestheticFramework,
    event: &SynestheticEvent,
) -> Result<Vec<SynestheticAction>, Box<dyn std::error::Error>> {
    match &event.data {
        EventData::Gesture { gesture_type, position, .. } => {
            match gesture_type.as_str() {
                "swipe_left" => {
                    Ok(vec![SynestheticAction::Visual {
                        target: "diffusion".to_string(),
                        parameter: "hue".to_string(),
                        value: -0.1,
                    }])
                }
                "swipe_right" => {
                    Ok(vec![SynestheticAction::Visual {
                        target: "diffusion".to_string(),
                        parameter: "hue".to_string(),
                        value: 0.1,
                    }])
                }
                "pinch" => {
                    Ok(vec![SynestheticAction::Audio {
                        target: "synth".to_string(),
                        parameter: "frequency".to_string(),
                        value: 0.5,
                    }])
                }
                _ => Ok(vec![])
            }
        }
        _ => Ok(vec![])
    }
}
```

## 🔧 Integration with Other Modules

### Module Interaction Diagram

```mermaid
graph TD
    A[Synesthetic Framework] --> B[Diffusion Models]
    A --> C[EEG Processing]
    A --> D[ONNX Runtime]
    A --> E[3D Model Generation]
    A --> F[Audio Synthesis]
    A --> G[Web Interface]
    
    B --> B1[Image Generation]
    B --> B2[Text Encoding]
    B --> B3[Model Inference]
    
    C --> C1[Signal Analysis]
    C --> C2[Frequency Bands]
    C --> C3[Feature Extraction]
    
    D --> D1[Model Loading]
    D --> D2[Hardware Acceleration]
    D --> D3[Inference Execution]
    
    E --> E1[Geometry Creation]
    E --> E2[Texture Mapping]
    E --> E3[Animation Processing]
    
    F --> F1[Waveform Generation]
    F --> F2[Audio Effects]
    F --> F3[Sound Synthesis]
    
    G --> G1[API Endpoints]
    G --> G2[WebSocket Streaming]
    G --> G3[UI Components]
    
    style A fill:#2196F3,stroke:#0D47A1
    style B fill:#9C27B0,stroke:#4A148C
    style C fill:#4CAF50,stroke:#388E3C
    style D fill:#FF9800,stroke:#E65100
    style E fill:#FF5722,stroke:#BF360C
    style F fill:#009688,stroke:#004D40
    style G fill:#795548,stroke:#3E2723
```

### Data Flow Between Modules

```mermaid
sequenceDiagram
    participant SF as Synesthetic Framework
    participant DM as Diffusion Models
    participant EP as EEG Processing
    participant OR as ONNX Runtime
    participant M3 as 3D Models
    participant AS as Audio Synthesis
    participant WI as Web Interface
    
    SF->>EP: Request EEG Analysis
    EP->>SF: Return Band Powers
    SF->>DM: Trigger Image Generation
    DM->>OR: Execute Model Inference
    OR->>DM: Return Image Data
    DM->>SF: Return Generated Image
    SF->>AS: Trigger Audio Synthesis
    AS->>SF: Return Audio Data
    SF->>M3: Request 3D Model
    M3->>SF: Return Model Data
    SF->>WI: Send Combined Results
```

## 🔄 Cross-Modal Interaction System

### Complete Cross-Modal Mapping Architecture

```mermaid
graph TD
    A[Sensory Modalities] --> B[Synesthetic Framework]
    B --> C[Feature Extraction]
    C --> D[Fusion Engine]
    D --> E[Cross-Modal Mapper]
    E --> F[Action Generator]
    F --> G[Output Systems]
    
    A1[Gesture Input] --> A
    A2[Visual Input] --> A
    A3[Audio Input] --> A
    A4[EEG Input] --> A
    A5[3D Model Input] --> A
    A6[Haptic Input] --> A
    A7[Thermal Input] --> A
    A8[Tactile Input] --> A
    
    C1[Position Data] --> C
    C2[Image Features] --> C
    C3[Audio Features] --> C
    C4[EEG Bands] --> C
    C5[Model Data] --> C
    C6[Force Data] --> C
    C7[Temperature Data] --> C
    C8[Pressure Data] --> C
    
    D1[Temporal Fusion] --> D
    D2[Spatial Fusion] --> D
    D3[Semantic Fusion] --> D
    D4[Contextual Fusion] --> D
    
    E1[Gesture → Visual] --> E
    E2[Audio → Visual] --> E
    E3[EEG → Audio] --> E
    E4[Visual → Audio] --> E
    E5[EEG → Visual] --> E
    E6[Audio → Haptic] --> E
    E7[Gesture → Audio] --> E
    E8[Visual → Haptic] --> E
    
    F1[Visual Actions] --> F
    F2[Audio Actions] --> F
    F3[3D Model Actions] --> F
    F4[Haptic Actions] --> F
    F5[EEG Feedback] --> F
    
    G1[Display Output] --> G
    G2[Audio Output] --> G
    G3[Haptic Output] --> G
    G4[File Output] --> G
    G5[Network Output] --> G
    
    style A fill:#FF9800,stroke:#E65100
    style B fill:#2196F3,stroke:#0D47A1
    style C fill:#4CAF50,stroke:#388E3C
    style D fill:#9C27B0,stroke:#4A148C
    style E fill:#FF5722,stroke:#BF360C
    style F fill:#009688,stroke:#004D40
    style G fill:#795548,stroke:#3E2723
```

### Cross-Modal Interaction Flow

```mermaid
sequenceDiagram
    participant Sensors as Hardware Sensors
    participant Framework as Synesthetic Framework
    participant Fusion as Fusion Engine
    participant Mapper as Cross-Modal Mapper
    participant Actions as Action Generator
    participant Outputs as Output Systems
    participant Feedback as User Feedback
    
    Sensors->>Framework: Raw Sensor Data
    Framework->>Framework: Preprocessing
    Framework->>Fusion: Extracted Features
    Fusion->>Fusion: Data Fusion
    Fusion->>Mapper: Fused Events
    Mapper->>Mapper: Cross-Modal Mapping
    Mapper->>Actions: Generated Actions
    Actions->>Outputs: Action Commands
    Outputs->>Feedback: System Response
    Feedback->>Sensors: Adaptive Feedback
```

### Cross-Modal Fusion Strategies

```mermaid
graph TD
    A[Input Modalities] --> B[Fusion Engine]
    B --> C[Temporal Fusion]
    B --> D[Spatial Fusion]
    B --> E[Semantic Fusion]
    B --> F[Contextual Fusion]
    
    A1[EEG Signals] --> A
    A2[Gesture Data] --> A
    A3[Audio Input] --> A
    A4[Visual Data] --> A
    
    C --> C1[Time-Series Alignment]
    C --> C2[Event Synchronization]
    
    D --> D1[Spatial Mapping]
    D --> D2[Coordinate Transformation]
    
    E --> E1[Feature Correlation]
    E --> E2[Semantic Embeddings]
    
    F --> F1[Context Analysis]
    F --> F2[Situation Awareness]
    
    C1 --> G[Unified Representation]
    C2 --> G
    D1 --> G
    D2 --> G
    E1 --> G
    E2 --> G
    F1 --> G
    F2 --> G
    
    G --> H[Cross-Modal Actions]
    
    style A fill:#FF9800,stroke:#E65100
    style B fill:#9C27B0,stroke:#4A148C
    style C fill:#2196F3,stroke:#0D47A1
    style D fill:#4CAF50,stroke:#388E3C
    style E fill:#FF5722,stroke:#BF360C
    style F fill:#009688,stroke:#004D40
    style G fill:#795548,stroke:#3E2723
    style H fill:#607D8B,stroke:#263238
```

### Real-time Cross-Modal Processing Pipeline

```mermaid
sequenceDiagram
    participant Input as Sensor Inputs
    participant Preproc as Preprocessing
    participant Extract as Feature Extraction
    participant Fuse as Fusion Engine
    participant Map as Cross-Modal Mapper
    participant Generate as Action Generator
    participant Output as Output Systems
    
    loop Real-time Processing
        Input->>Preproc: Raw Data Streams
        Preproc->>Extract: Cleaned Signals
        Extract->>Fuse: Feature Vectors
        Fuse->>Map: Fused Events
        Map->>Generate: Mapped Actions
        Generate->>Output: Control Commands
        Output-->>Input: Feedback Signals
    end
```