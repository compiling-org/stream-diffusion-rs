## 🔄 WebSocket API

Real-time data streaming is available through WebSocket connections.

### Connection
```
ws://localhost:3000/ws
```

### Messages

#### Client to Server
```json
{
  "type": "subscribe",
  "topic": "eeg_data"
}
```

#### Server to Client
```json
{
  "type": "eeg_data",
  "data": {
    "timestamp": "2025-11-16T10:30:45Z",
    "channels": [0.5, 0.6, 0.4, 0.7, 0.3, 0.8],
    "bands": {
      "alpha": 0.7,
      "beta": 0.3,
      "theta": 0.6,
      "delta": 0.2,
      "gamma": 0.1
    }
  }
}
```

### WebSocket Communication Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    participant F as Synesthetic Framework
    participant M as ML Models
    
    C->>S: WebSocket Connection
    S->>C: Connection Acknowledged
    C->>S: Subscribe to eeg_data
    S->>F: Register Subscription
    F->>M: Process EEG Data
    M->>F: Generate Events
    F->>S: Broadcast Events
    S->>C: Send EEG Data
    C->>S: Send Gesture Data
    S->>F: Process Gesture
    F->>F: Cross-Modal Mapping
    F->>S: Generate Actions
    S->>C: Send Visual Updates
```

### Real-time Data Pipeline

```mermaid
graph TD
    A[WebSocket Client] --> B[WebSocket Server]
    B --> C[Event Router]
    C --> D[Sensory Processing]
    D --> E[Fusion Engine]
    E --> F[Action Generator]
    F --> G[Response Handler]
    G --> H[Client Notification]
    H --> A
    
    style A fill:#4CAF50,stroke:#388E3C
    style B fill:#2196F3,stroke:#0D47A1
    style C fill:#FF9800,stroke:#E65100
    style D fill:#9C27B0,stroke:#4A148C
    style E fill:#4CAF50,stroke:#388E3C
    style F fill:#FF5722,stroke:#BF360C
    style G fill:#009688,stroke:#004D40
    style H fill:#795548,stroke:#3E2723
```

## 🌐 Cross-Modal API Integration

### Complete API Integration Architecture

```mermaid
graph TD
    A[External Applications] --> B[API Gateway]
    B --> C[Authentication Layer]
    C --> D[Request Router]
    D --> E[Synesthetic API]
    D --> F[Diffusion API]
    D --> G[EEG API]
    D --> H[3D Model API]
    D --> I[Audio API]
    D --> J[System API]
    
    E --> E1[Input Processing]
    E --> E2[Modality Control]
    E --> E3[Fusion Operations]
    E --> E4[Mapping Services]
    
    F --> F1[Image Generation]
    F --> F2[Model Management]
    
    G --> G1[Signal Analysis]
    G --> G2[Feature Extraction]
    
    H --> H1[Model Generation]
    H --> H2[Model Management]
    
    I --> I1[Audio Synthesis]
    I --> I2[Audio Management]
    
    J --> J1[System Status]
    J --> J2[Configuration]
    
    E1 --> K[Synesthetic Framework]
    E2 --> K
    E3 --> K
    E4 --> K
    
    F1 --> L[Diffusion Engine]
    F2 --> M[Model Registry]
    
    G1 --> N[EEG Processor]
    G2 --> N
    
    H1 --> O[3D Generator]
    H2 --> P[Model Manager]
    
    I1 --> Q[Audio Synthesizer]
    I2 --> R[Audio Manager]
    
    J1 --> S[System Monitor]
    J2 --> T[Config Manager]
    
    K --> U[Cross-Modal Mapping]
    K --> V[Fusion Processing]
    K --> W[Action Generation]
    
    style A fill:#4CAF50,stroke:#388E3C
    style B fill:#2196F3,stroke:#0D47A1
    style C fill:#FF9800,stroke:#E65100
    style D fill:#9C27B0,stroke:#4A148C
    style E fill:#4CAF50,stroke:#388E3C
    style F fill:#FF5722,stroke:#BF360C
    style G fill:#009688,stroke:#004D40
    style H fill:#795548,stroke:#3E2723
    style I fill:#607D8B,stroke:#263238
    style J fill:#E91E63,stroke:#880E4F
    style K fill:#9C27B0,stroke:#4A148C
```

### Cross-Modal API Data Flow

```mermaid
sequenceDiagram
    participant Client as External Client
    participant API as API Gateway
    participant Router as Request Router
    participant SynAPI as Synesthetic API
    participant Framework as Synesthetic Framework
    participant Modules as Backend Modules
    participant Outputs as Output Systems
    
    Client->>API: POST /api/synesthetic/process
    API->>Router: Route Request
    Router->>SynAPI: Handle Synesthetic Request
    SynAPI->>Framework: Process Input
    Framework->>Framework: Feature Extraction
    Framework->>Framework: Fusion Processing
    Framework->>Framework: Cross-Modal Mapping
    Framework->>Modules: Generate Actions
    Modules->>Outputs: Execute Actions
    Outputs->>Framework: Return Results
    Framework->>SynAPI: Send Response
    SynAPI->>Router: Return Results
    Router->>API: Send Response
    API->>Client: HTTP Response
```

### API Endpoint Classification

```mermaid
graph TD
    A[API Endpoints] --> B[Control APIs]
    A --> C[Data Streaming APIs]
    A --> D[Model Management APIs]
    A --> E[Analysis APIs]
    
    B --> B1[Generation Control]
    B --> B2[Processing Control]
    B --> B3[System Control]
    
    C --> C1[WebSocket Streams]
    C --> C2[Event Feeds]
    C --> C3[Real-time Updates]
    
    D --> D1[Model Upload]
    D --> D2[Model Registry]
    D --> D3[Model Inference]
    
    E --> E1[EEG Analysis]
    E --> E2[Signal Processing]
    E --> E3[Data Visualization]
    
    B1 --> F[Core Functionality]
    B2 --> F
    B3 --> F
    C1 --> F
    C2 --> F
    C3 --> F
    D1 --> F
    D2 --> F
    D3 --> F
    E1 --> F
    E2 --> F
    E3 --> F
    
    F --> G[Client Applications]
    
    style A fill:#2196F3,stroke:#0D47A1
    style B fill:#4CAF50,stroke:#388E3C
    style C fill:#FF9800,stroke:#E65100
    style D fill:#9C27B0,stroke:#4A148C
    style E fill:#009688,stroke:#004D40
    style F fill:#FF5722,stroke:#BF360C
    style G fill:#795548,stroke:#3E2723
```

### Real-time API Performance Metrics

```mermaid
graph LR
    A[API Requests] --> B[Request Processing]
    B --> C[Response Generation]
    C --> D[Client Delivery]
    
    A1[High Priority] --> A
    A2[Real-time Streams] --> A
    A3[Batch Requests] --> A
    
    B1[Authentication] --> B
    B2[Validation] --> B
    B3[Routing] --> B
    B4[Execution] --> B
    
    C1[Data Serialization] --> C
    C2[Response Formatting] --> C
    C3[Compression] --> C
    
    D1[Network Transmission] --> D
    D2[Client Reception] --> D
    D3[Acknowledgement] --> D
    
    style A fill:#2196F3,stroke:#0D47A1
    style B fill:#4CAF50,stroke:#388E3C
    style C fill:#FF9800,stroke:#E65100
    style D fill:#9C27B0,stroke:#4A148C
```