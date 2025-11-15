//! Synesthetic Framework Demo
//! 
//! This example demonstrates how to use the synesthetic framework to create
//! a multimodal AI experience that connects gesture, vision, audio, and other
//! sensory modalities in a unified experience.

use stream_diffusion_rs::synesthesia::*;
use stream_diffusion_rs::gesture::*;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Synesthetic Framework Demo");
    println!("============================");
    
    // Create synesthetic framework
    let mut framework = SynestheticFramework::new()?;
    println!("✅ Created synesthetic framework");
    
    // Create gesture system
    let gesture_system = GestureSystem::new()?;
    println!("✅ Created gesture system");
    
    // Create synesthetic controller
    let mut controller = SynestheticController::new(framework)?;
    println!("✅ Created synesthetic controller");
    
    // Simulate gesture events
    println!("\n🎮 Simulating gesture events...");
    
    // Create sample gesture events
    let gesture_events = vec![
        SynestheticEvent {
            modality: SensoryModality::Gesture,
            data: EventData::Gesture {
                gesture_type: "swipe_left".to_string(),
                position: Some((0.3, 0.5, 0.0)),
                velocity: Some((-0.2, 0.0, 0.0)),
                hand_id: Some(1),
            },
            timestamp: chrono::Utc::now(),
            confidence: 0.92,
            context: {
                let mut ctx = std::collections::HashMap::new();
                ctx.insert("user_id".to_string(), "demo_user".to_string());
                ctx
            },
        },
        SynestheticEvent {
            modality: SensoryModality::Gesture,
            data: EventData::Gesture {
                gesture_type: "pinch".to_string(),
                position: Some((0.7, 0.3, 0.1)),
                velocity: Some((0.0, 0.0, 0.0)),
                hand_id: Some(1),
            },
            timestamp: chrono::Utc::now(),
            confidence: 0.88,
            context: {
                let mut ctx = std::collections::HashMap::new();
                ctx.insert("user_id".to_string(), "demo_user".to_string());
                ctx
            },
        },
        SynestheticEvent {
            modality: SensoryModality::Gesture,
            data: EventData::Gesture {
                gesture_type: "victory".to_string(),
                position: Some((0.8, 0.7, 0.0)),
                velocity: Some((0.1, 0.1, 0.0)),
                hand_id: Some(2),
            },
            timestamp: chrono::Utc::now(),
            confidence: 0.95,
            context: {
                let mut ctx = std::collections::HashMap::new();
                ctx.insert("user_id".to_string(), "demo_user".to_string());
                ctx
            },
        },
    ];
    
    // Process gesture events
    let actions = controller.process_events(gesture_events).await?;
    println!("✅ Processed {} gesture events", actions.len());
    
    // Execute actions
    controller.execute_actions().await?;
    println!("✅ Executed synesthetic actions");
    
    // Simulate visual events
    println!("\n👁️  Simulating visual events...");
    
    let visual_events = vec![
        SynestheticEvent {
            modality: SensoryModality::Vision,
            data: EventData::Visual {
                features: vec![0.5, 0.3, 0.8, 0.2],
                objects: vec![
                    VisualObject {
                        id: "obj1".to_string(),
                        class: "person".to_string(),
                        bounding_box: (0.2, 0.3, 0.4, 0.6),
                        confidence: 0.91,
                        features: vec![0.1, 0.9, 0.2, 0.8],
                    }
                ],
                scene_description: "A person standing in a colorful room".to_string(),
            },
            timestamp: chrono::Utc::now(),
            confidence: 0.89,
            context: std::collections::HashMap::new(),
        }
    ];
    
    // Process visual events
    let actions = controller.process_events(visual_events).await?;
    println!("✅ Processed {} visual events", actions.len());
    
    // Execute actions
    controller.execute_actions().await?;
    println!("✅ Executed synesthetic actions");
    
    // Simulate audio events
    println!("\n🎵 Simulating audio events...");
    
    let audio_events = vec![
        SynestheticEvent {
            modality: SensoryModality::Audio,
            data: EventData::Audio {
                features: vec![0.7, 0.4, 0.6, 0.9],
                frequency_bands: {
                    let mut bands = std::collections::HashMap::new();
                    bands.insert("bass".to_string(), 0.8);
                    bands.insert("mid".to_string(), 0.5);
                    bands.insert("treble".to_string(), 0.3);
                    bands
                },
                rhythm: Some(RhythmPattern {
                    tempo: 120.0,
                    beats: vec![0.0, 0.25, 0.5, 0.75],
                    signature: "4/4".to_string(),
                }),
            },
            timestamp: chrono::Utc::now(),
            confidence: 0.93,
            context: std::collections::HashMap::new(),
        }
    ];
    
    // Process audio events
    let actions = controller.process_events(audio_events).await?;
    println!("✅ Processed {} audio events", actions.len());
    
    // Execute actions
    controller.execute_actions().await?;
    println!("✅ Executed synesthetic actions");
    
    // Simulate EEG events
    println!("\n🧠 Simulating EEG events...");
    
    let eeg_events = vec![
        SynestheticEvent {
            modality: SensoryModality::EEG,
            data: EventData::EEG {
                band_powers: {
                    let mut powers = std::collections::HashMap::new();
                    powers.insert("alpha".to_string(), 0.75);
                    powers.insert("beta".to_string(), 0.65);
                    powers.insert("theta".to_string(), 0.45);
                    powers.insert("delta".to_string(), 0.30);
                    powers
                },
                connectivity: vec![0.8, 0.6, 0.7, 0.5, 0.9],
                emotional_state: Some("focused".to_string()),
            },
            timestamp: chrono::Utc::now(),
            confidence: 0.87,
            context: std::collections::HashMap::new(),
        }
    ];
    
    // Process EEG events
    let actions = controller.process_events(eeg_events).await?;
    println!("✅ Processed {} EEG events", actions.len());
    
    // Execute actions
    controller.execute_actions().await?;
    println!("✅ Executed synesthetic actions");
    
    // Demonstrate fusion
    println!("\n🔄 Demonstrating sensory fusion...");
    
    // Create events from different modalities for fusion
    let fusion_events = vec![
        SynestheticEvent {
            modality: SensoryModality::Gesture,
            data: EventData::Gesture {
                gesture_type: "swipe_up".to_string(),
                position: Some((0.5, 0.5, 0.0)),
                velocity: Some((0.0, 0.3, 0.0)),
                hand_id: Some(1),
            },
            timestamp: chrono::Utc::now(),
            confidence: 0.90,
            context: std::collections::HashMap::new(),
        },
        SynestheticEvent {
            modality: SensoryModality::Audio,
            data: EventData::Audio {
                features: vec![0.6, 0.7, 0.8, 0.9],
                frequency_bands: {
                    let mut bands = std::collections::HashMap::new();
                    bands.insert("bass".to_string(), 0.9);
                    bands.insert("mid".to_string(), 0.7);
                    bands.insert("treble".to_string(), 0.5);
                    bands
                },
                rhythm: None,
            },
            timestamp: chrono::Utc::now(),
            confidence: 0.85,
            context: std::collections::HashMap::new(),
        }
    ];
    
    // Fuse events using different strategies
    let fused_temporal = controller.framework.fuse_events(fusion_events.clone(), FusionType::Temporal).await?;
    println!("✅ Temporal fusion completed");
    
    let fused_semantic = controller.framework.fuse_events(fusion_events.clone(), FusionType::Semantic).await?;
    println!("✅ Semantic fusion completed");
    
    let fused_contextual = controller.framework.fuse_events(fusion_events, FusionType::Contextual).await?;
    println!("✅ Contextual fusion completed");
    
    // Show fused results
    println!("\n📊 Fusion Results:");
    println!("  Temporal: {:?} event with {:.2} confidence", fused_temporal.modality, fused_temporal.confidence);
    println!("  Semantic: {:?} event with {:.2} confidence", fused_semantic.modality, fused_semantic.confidence);
    println!("  Contextual: {:?} event with {:.2} confidence", fused_contextual.modality, fused_contextual.confidence);
    
    // Demonstrate cross-modal mapping
    println!("\n🔗 Demonstrating cross-modal mappings...");
    
    // In a real implementation, this would show how gestures map to visual/audio changes
    println!("  Swipe Left → Hue Shift: -10%");
    println!("  Pinch → Audio Frequency: 440Hz");
    println!("  Victory Sign → 3D Rotation: 15°");
    println!("  Alpha Waves ↑ → Visual Brightness: +20%");
    
    println!("\n🎉 Synesthetic demo completed successfully!");
    println!("\n💡 The synesthetic framework enables:");
    println!("   • Real-time multimodal AI experiences");
    println!("   • Cross-sensory artistic expression");
    println!("   • Brain-computer interface applications");
    println!("   • Immersive interactive installations");
    println!("   • Therapeutic and educational tools");
    
    Ok(())
}