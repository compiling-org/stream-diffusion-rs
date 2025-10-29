//! Real-Time Data Streaming Protocols Framework
//!
//! Advanced real-time data streaming protocols for neurophysiological data
//! with low-latency transmission, quality-of-service guarantees, and adaptive streaming.
//! Adapted from Neuro-Emotive AI's streaming framework for Stream Diffusion.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, broadcast, Semaphore};
use tokio::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// Real-Time Streaming Framework
pub struct RealTimeStreamingFramework {
    /// Stream managers for different data types
    stream_managers: HashMap<String, Arc<StreamManager>>,

    /// Quality of Service (QoS) policies
    qos_policies: HashMap<String, QoSPolicy>,

    /// Adaptive streaming controllers
    adaptive_controllers: HashMap<String, AdaptiveStreamingController>,

    /// Stream monitoring system
    monitoring_system: StreamMonitoringSystem,

    /// Configuration
    config: StreamingConfig,
}

impl RealTimeStreamingFramework {
    /// Create a new real-time streaming framework
    pub fn new(config: StreamingConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let mut framework = Self {
            stream_managers: HashMap::new(),
            qos_policies: HashMap::new(),
            adaptive_controllers: HashMap::new(),
            monitoring_system: StreamMonitoringSystem::new()?,
            config,
        };

        // Initialize default stream managers
        framework.initialize_stream_managers()?;

        // Initialize QoS policies
        framework.initialize_qos_policies()?;

        Ok(framework)
    }

    /// Create a new data stream
    pub async fn create_stream(&mut self, stream_id: &str, config: StreamConfiguration)
        -> Result<StreamHandle, Box<dyn std::error::Error>> {

        let manager = Arc::new(StreamManager::new(stream_id.to_string(), config.clone()).await?);

        // Create adaptive controller
        let controller = AdaptiveStreamingController::new(stream_id.to_string(), config).await?;
        self.adaptive_controllers.insert(stream_id.to_string(), controller);

        self.stream_managers.insert(stream_id.to_string(), manager.clone());

        Ok(StreamHandle {
            stream_id: stream_id.to_string(),
            manager,
        })
    }

    /// Send data to a stream
    pub async fn send_to_stream(&mut self, stream_id: &str, data: StreamData)
        -> Result<(), Box<dyn std::error::Error>> {

        let manager = self.stream_managers.get(stream_id)
            .ok_or_else(|| format!("Stream {} not found", stream_id))?;

        // Check QoS policy
        if let Some(policy) = self.qos_policies.get(stream_id) {
            if !policy.check_compliance(&data)? {
                return Err("QoS policy violation".into());
            }
        }

        // Get size before moving data
        let size_bytes = data.size_bytes;

        // Send data through manager
        manager.send_data(data).await?;

        // Update monitoring
        self.monitoring_system.record_data_point(stream_id, size_bytes).await?;

        Ok(())
    }

    /// Subscribe to a stream
    pub async fn subscribe_to_stream(&self, stream_id: &str)
        -> Result<StreamSubscription, Box<dyn std::error::Error>> {

        let manager = self.stream_managers.get(stream_id)
            .ok_or_else(|| format!("Stream {} not found", stream_id))?;

        let subscription = manager.create_subscription().await?;

        Ok(subscription)
    }

    /// Get stream statistics
    pub async fn get_stream_stats(&self, stream_id: &str)
        -> Result<StreamStatistics, Box<dyn std::error::Error>> {

        let manager = self.stream_managers.get(stream_id)
            .ok_or_else(|| format!("Stream {} not found", stream_id))?;

        let stats = manager.get_statistics().await?;
        Ok(stats)
    }

    /// Get overall streaming health
    pub async fn get_streaming_health(&self) -> Result<StreamingHealth, Box<dyn std::error::Error>> {
        let mut total_streams = 0;
        let mut healthy_streams = 0;
        let mut total_throughput = 0.0;
        let mut total_latency = 0.0;

        for manager in self.stream_managers.values() {
            total_streams += 1;
            let stats = manager.get_statistics().await?;

            if stats.is_healthy() {
                healthy_streams += 1;
            }

            total_throughput += stats.throughput_mbps;
            total_latency += stats.average_latency_ms;
        }

        let average_latency = if total_streams > 0 {
            total_latency / total_streams as f32
        } else {
            0.0
        };

        let overall_health = if healthy_streams == total_streams && total_streams > 0 {
            StreamHealthStatus::Healthy
        } else if healthy_streams > 0 {
            StreamHealthStatus::Degraded
        } else {
            StreamHealthStatus::Unhealthy
        };

        Ok(StreamingHealth {
            overall_status: overall_health,
            total_streams,
            healthy_streams,
            average_throughput_mbps: total_throughput,
            average_latency_ms: average_latency,
            assessment_time: Utc::now(),
        })
    }

    /// Adjust streaming parameters based on network conditions
    pub async fn adapt_streaming(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for (stream_id, controller) in &mut self.adaptive_controllers {
            let network_conditions = self.monitoring_system.get_network_conditions(stream_id).await?;
            controller.adapt_to_conditions(network_conditions).await?;
        }

        Ok(())
    }

    fn initialize_stream_managers(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Initialize diffusion stream
        let diffusion_config = StreamConfiguration {
            data_type: "diffusion_frames".to_string(),
            max_bandwidth_mbps: 100.0,
            target_latency_ms: 33.0, // ~30 FPS
            buffer_size_packets: 30,
            compression_enabled: true,
            encryption_enabled: false,
            priority: StreamPriority::High,
        };

        // Initialize EEG stream
        let eeg_config = StreamConfiguration {
            data_type: "eeg".to_string(),
            max_bandwidth_mbps: 50.0,
            target_latency_ms: 5.0,
            buffer_size_packets: 1000,
            compression_enabled: true,
            encryption_enabled: true,
            priority: StreamPriority::High,
        };

        // Initialize audio stream
        let audio_config = StreamConfiguration {
            data_type: "audio".to_string(),
            max_bandwidth_mbps: 20.0,
            target_latency_ms: 10.0,
            buffer_size_packets: 500,
            compression_enabled: true,
            encryption_enabled: false,
            priority: StreamPriority::Medium,
        };

        Ok(())
    }

    fn initialize_qos_policies(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Diffusion QoS Policy - real-time visual streaming
        let diffusion_policy = QoSPolicy {
            max_latency_ms: 50.0,
            min_throughput_mbps: 80.0,
            max_jitter_ms: 5.0,
            max_packet_loss_percent: 0.5,
            priority: StreamPriority::High,
        };
        self.qos_policies.insert("diffusion_frames".to_string(), diffusion_policy);

        // EEG QoS Policy - strict latency requirements
        let eeg_policy = QoSPolicy {
            max_latency_ms: 10.0,
            min_throughput_mbps: 40.0,
            max_jitter_ms: 2.0,
            max_packet_loss_percent: 0.1,
            priority: StreamPriority::High,
        };
        self.qos_policies.insert("eeg".to_string(), eeg_policy);

        // Audio QoS Policy - moderate requirements
        let audio_policy = QoSPolicy {
            max_latency_ms: 20.0,
            min_throughput_mbps: 15.0,
            max_jitter_ms: 5.0,
            max_packet_loss_percent: 1.0,
            priority: StreamPriority::Medium,
        };
        self.qos_policies.insert("audio".to_string(), audio_policy);

        Ok(())
    }
}

/// Stream Handle
pub struct StreamHandle {
    pub stream_id: String,
    manager: Arc<StreamManager>,
}

impl StreamHandle {
    pub async fn close(self) -> Result<(), Box<dyn std::error::Error>> {
        self.manager.close().await
    }
}

/// Stream Subscription
pub struct StreamSubscription {
    receiver: broadcast::Receiver<StreamData>,
}

impl StreamSubscription {
    /// Receive next data packet
    pub async fn recv(&mut self) -> Result<StreamData, Box<dyn std::error::Error>> {
        Ok(self.receiver.recv().await?)
    }
}

/// Stream Data
#[derive(Debug, Clone)]
pub struct StreamData {
    pub timestamp: DateTime<Utc>,
    pub sequence_number: u64,
    pub data_type: String,
    pub payload: Vec<u8>,
    pub size_bytes: usize,
    pub quality_score: f32,
    pub metadata: HashMap<String, String>,
}

/// Stream Configuration
#[derive(Debug, Clone)]
pub struct StreamConfiguration {
    pub data_type: String,
    pub max_bandwidth_mbps: f32,
    pub target_latency_ms: f32,
    pub buffer_size_packets: usize,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
    pub priority: StreamPriority,
}

/// Stream Priority
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Streaming Configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub max_concurrent_streams: usize,
    pub default_buffer_size: usize,
    pub enable_compression: bool,
    pub enable_encryption: bool,
    pub monitoring_interval_ms: u64,
    pub adaptive_streaming_enabled: bool,
}

/// QoS Policy
#[derive(Debug, Clone)]
pub struct QoSPolicy {
    pub max_latency_ms: f32,
    pub min_throughput_mbps: f32,
    pub max_jitter_ms: f32,
    pub max_packet_loss_percent: f32,
    pub priority: StreamPriority,
}

impl QoSPolicy {
    pub fn check_compliance(&self, data: &StreamData) -> Result<bool, Box<dyn std::error::Error>> {
        // In real implementation, would check actual latency, throughput, etc.
        // For now, assume compliance
        Ok(true)
    }
}

/// Stream Statistics
#[derive(Debug, Clone)]
pub struct StreamStatistics {
    pub stream_id: String,
    pub total_packets_sent: u64,
    pub total_packets_received: u64,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub throughput_mbps: f32,
    pub average_latency_ms: f32,
    pub jitter_ms: f32,
    pub packet_loss_percent: f32,
    pub buffer_usage_percent: f32,
    pub last_activity: DateTime<Utc>,
}

impl StreamStatistics {
    pub fn is_healthy(&self) -> bool {
        self.packet_loss_percent < 5.0 &&
        self.average_latency_ms < 50.0 &&
        self.buffer_usage_percent < 90.0
    }
}

/// Streaming Health
#[derive(Debug, Clone)]
pub struct StreamingHealth {
    pub overall_status: StreamHealthStatus,
    pub total_streams: usize,
    pub healthy_streams: usize,
    pub average_throughput_mbps: f32,
    pub average_latency_ms: f32,
    pub assessment_time: DateTime<Utc>,
}

/// Stream Health Status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamHealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Stream Manager
pub struct StreamManager {
    stream_id: String,
    config: StreamConfiguration,
    sender: broadcast::Sender<StreamData>,
    buffer: Arc<Mutex<VecDeque<StreamData>>>,
    stats: Arc<Mutex<StreamStatistics>>,
    is_active: Arc<Mutex<bool>>,
}

impl StreamManager {
    pub async fn new(stream_id: String, config: StreamConfiguration)
        -> Result<Self, Box<dyn std::error::Error>> {

        let (sender, _) = broadcast::channel(config.buffer_size_packets);
        let buffer = Arc::new(Mutex::new(VecDeque::with_capacity(config.buffer_size_packets)));

        let stats = Arc::new(Mutex::new(StreamStatistics {
            stream_id: stream_id.clone(),
            total_packets_sent: 0,
            total_packets_received: 0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
            throughput_mbps: 0.0,
            average_latency_ms: 0.0,
            jitter_ms: 0.0,
            packet_loss_percent: 0.0,
            buffer_usage_percent: 0.0,
            last_activity: Utc::now(),
        }));

        Ok(Self {
            stream_id,
            config,
            sender,
            buffer,
            stats,
            is_active: Arc::new(Mutex::new(true)),
        })
    }

    pub async fn send_data(&self, data: StreamData) -> Result<(), Box<dyn std::error::Error>> {
        let is_active = *self.is_active.lock().unwrap();
        if !is_active {
            return Err("Stream is not active".into());
        }

        // Add to buffer
        {
            let mut buffer = self.buffer.lock().unwrap();
            if buffer.len() >= self.config.buffer_size_packets {
                buffer.pop_front(); // Remove oldest
            }
            buffer.push_back(data.clone());
        }

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_packets_sent += 1;
            stats.total_bytes_sent += data.size_bytes as u64;
            stats.last_activity = Utc::now();

            // Calculate buffer usage
            let buffer_len = self.buffer.lock().unwrap().len();
            stats.buffer_usage_percent = (buffer_len as f32 / self.config.buffer_size_packets as f32) * 100.0;
        }

        // Broadcast to subscribers
        let _ = self.sender.send(data);

        Ok(())
    }

    pub async fn create_subscription(&self) -> Result<StreamSubscription, Box<dyn std::error::Error>> {
        let receiver = self.sender.subscribe();

        Ok(StreamSubscription { receiver })
    }

    pub async fn get_statistics(&self) -> Result<StreamStatistics, Box<dyn std::error::Error>> {
        Ok(self.stats.lock().unwrap().clone())
    }

    pub async fn close(&self) -> Result<(), Box<dyn std::error::Error>> {
        *self.is_active.lock().unwrap() = false;
        Ok(())
    }
}

/// Adaptive Streaming Controller
pub struct AdaptiveStreamingController {
    stream_id: String,
    config: StreamConfiguration,
    current_bitrate: f32,
    adaptation_history: VecDeque<AdaptationDecision>,
}

impl AdaptiveStreamingController {
    pub async fn new(stream_id: String, config: StreamConfiguration)
        -> Result<Self, Box<dyn std::error::Error>> {

        let current_bitrate = config.max_bandwidth_mbps;
        Ok(Self {
            stream_id,
            config,
            current_bitrate,
            adaptation_history: VecDeque::with_capacity(50),
        })
    }

    pub async fn adapt_to_conditions(&mut self, conditions: NetworkConditions)
        -> Result<(), Box<dyn std::error::Error>> {

        let decision = self.make_adaptation_decision(&conditions).await?;

        // Apply decision
        match decision.action {
            AdaptationAction::IncreaseBitrate => {
                self.current_bitrate = (self.current_bitrate * 1.2).min(self.config.max_bandwidth_mbps);
            }
            AdaptationAction::DecreaseBitrate => {
                self.current_bitrate = (self.current_bitrate * 0.8).max(1.0);
            }
            AdaptationAction::MaintainBitrate => {}
        }

        // Record decision
        self.adaptation_history.push_back(decision);
        if self.adaptation_history.len() > 50 {
            self.adaptation_history.pop_front();
        }

        Ok(())
    }

    async fn make_adaptation_decision(&self, conditions: &NetworkConditions)
        -> Result<AdaptationDecision, Box<dyn std::error::Error>> {

        let action = if conditions.bandwidth_mbps < self.current_bitrate * 0.8 {
            AdaptationAction::DecreaseBitrate
        } else if conditions.bandwidth_mbps > self.current_bitrate * 1.5 && conditions.latency_ms < self.config.target_latency_ms {
            AdaptationAction::IncreaseBitrate
        } else {
            AdaptationAction::MaintainBitrate
        };

        let action_clone = action.clone();
        Ok(AdaptationDecision {
            timestamp: Utc::now(),
            conditions: conditions.clone(),
            action,
            new_bitrate: match action_clone {
                AdaptationAction::IncreaseBitrate => (self.current_bitrate * 1.2).min(self.config.max_bandwidth_mbps),
                AdaptationAction::DecreaseBitrate => (self.current_bitrate * 0.8).max(1.0),
                AdaptationAction::MaintainBitrate => self.current_bitrate,
            },
            reason: format!("Network conditions: {:.1} Mbps, {:.1} ms latency", conditions.bandwidth_mbps, conditions.latency_ms),
        })
    }
}

/// Network Conditions
#[derive(Debug, Clone)]
pub struct NetworkConditions {
    pub bandwidth_mbps: f32,
    pub latency_ms: f32,
    pub jitter_ms: f32,
    pub packet_loss_percent: f32,
}

/// Adaptation Decision
#[derive(Debug, Clone)]
pub struct AdaptationDecision {
    pub timestamp: DateTime<Utc>,
    pub conditions: NetworkConditions,
    pub action: AdaptationAction,
    pub new_bitrate: f32,
    pub reason: String,
}

/// Adaptation Action
#[derive(Debug, Clone)]
pub enum AdaptationAction {
    IncreaseBitrate,
    DecreaseBitrate,
    MaintainBitrate,
}

/// Stream Monitoring System
pub struct StreamMonitoringSystem {
    stream_metrics: HashMap<String, StreamMetrics>,
    network_monitor: NetworkMonitor,
}

impl StreamMonitoringSystem {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            stream_metrics: HashMap::new(),
            network_monitor: NetworkMonitor::new()?,
        })
    }

    pub async fn record_data_point(&mut self, stream_id: &str, data_size: usize)
        -> Result<(), Box<dyn std::error::Error>> {

        let metrics = self.stream_metrics.entry(stream_id.to_string())
            .or_insert_with(|| StreamMetrics::new(stream_id.to_string()));

        metrics.record_data_point(data_size).await?;
        Ok(())
    }

    pub async fn get_network_conditions(&self, stream_id: &str)
        -> Result<NetworkConditions, Box<dyn std::error::Error>> {

        // In real implementation, would measure actual network conditions
        Ok(NetworkConditions {
            bandwidth_mbps: 100.0, // Placeholder
            latency_ms: 5.0,
            jitter_ms: 1.0,
            packet_loss_percent: 0.1,
        })
    }
}

/// Stream Metrics
pub struct StreamMetrics {
    stream_id: String,
    data_points: VecDeque<DataPoint>,
    throughput_calculator: ThroughputCalculator,
}

impl StreamMetrics {
    pub fn new(stream_id: String) -> Self {
        Self {
            stream_id,
            data_points: VecDeque::with_capacity(1000),
            throughput_calculator: ThroughputCalculator::new(),
        }
    }

    pub async fn record_data_point(&mut self, data_size: usize) -> Result<(), Box<dyn std::error::Error>> {
        let point = DataPoint {
            timestamp: Instant::now(),
            size_bytes: data_size,
        };

        self.data_points.push_back(point);
        if self.data_points.len() > 1000 {
            self.data_points.pop_front();
        }

        self.throughput_calculator.add_data_point(data_size).await?;

        Ok(())
    }
}

/// Data Point
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub timestamp: Instant,
    pub size_bytes: usize,
}

/// Throughput Calculator
pub struct ThroughputCalculator {
    data_sizes: VecDeque<(Instant, usize)>,
    total_bytes: usize,
    window_duration: Duration,
}

impl ThroughputCalculator {
    pub fn new() -> Self {
        Self {
            data_sizes: VecDeque::with_capacity(100),
            total_bytes: 0,
            window_duration: Duration::from_secs(1),
        }
    }

    pub async fn add_data_point(&mut self, size: usize) -> Result<(), Box<dyn std::error::Error>> {
        let now = Instant::now();
        self.data_sizes.push_back((now, size));
        self.total_bytes += size;

        // Remove old data points outside the window
        while let Some(&(timestamp, old_size)) = self.data_sizes.front() {
            if now.duration_since(timestamp) > self.window_duration {
                self.data_sizes.pop_front();
                self.total_bytes -= old_size;
            } else {
                break;
            }
        }

        Ok(())
    }

    pub fn get_throughput_mbps(&self) -> f32 {
        if self.data_sizes.is_empty() {
            return 0.0;
        }

        let window_bytes = self.total_bytes as f32;
        let window_seconds = self.window_duration.as_secs_f32();
        let bits_per_second = (window_bytes * 8.0) / window_seconds;
        bits_per_second / 1_000_000.0 // Convert to Mbps
    }
}

/// Network Monitor
pub struct NetworkMonitor;

impl NetworkMonitor {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self)
    }
}