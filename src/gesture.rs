use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono;

/// Recognized gesture types
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum RecognizedGesture {
    Relaxed,
    Focused,
    Meditation,
    Stress,
    Calm,
    Excited,
    Tired,
    Confused,
    // Additional gestures for different input sources
    SwipeLeft,
    SwipeRight,
    SwipeUp,
    SwipeDown,
    Pinch,
    Grab,
    Point,
    Fist,
    OpenHand,
    Victory,
    ThumbsUp,
    ThumbsDown,
}

/// Gesture input data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureInput {
    pub gesture_type: String,
    pub position: (f32, f32),
    pub velocity: (f32, f32),
    pub hand_id: Option<u32>,
    pub confidence: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}