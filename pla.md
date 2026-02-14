# Project Plan: OpenMOSS - Distributed Voice Agent System

## Executive Summary

OpenMOSS is a self-hosted, distributed voice assistant system designed for home deployment. It combines edge-based voice activation with a centralized AI agent runtime, utilizing local Large Language Models (LLM), on-device Speech-to-Text (STT), and local Text-to-Speech (TTS) synthesis. The system is architected for privacy (no cloud dependency), cost-efficiency (minimal ongoing costs), and extensibility (modular agent workflows via LangGraph).

---

## 1. System Architecture Overview

### 1.1 High-Level Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENDPOINT LAYER                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Room A    │  │   Room B    │  │   Room C    │  │   Room N    │         │
│  │  (Pi Zero)  │  │  (Pi Zero)  │  │  (ESP32-S3) │  │  (Variable) │         │
│  │             │  │             │  │             │  │             │         │
│  │ - Wake Word │  │ - Wake Word │  │ - Wake Word │  │ - Wake Word │         │
│  │ - VAD       │  │ - VAD       │  │ - VAD (lite)│  │ - VAD       │         │
│  │ - Opus Enc  │  │ - Opus Enc  │  │ - PCM Raw   │  │ - Stream    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            NETWORK LAYER                                    │
│  • WiFi 6 (backbone)  • MQTT (control plane)  • WebSocket/HTTP (data plane) │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                             SERVER LAYER                                      │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                        FastAPI API Gateway                             │   │
│  │  - Authentication  - Rate Limiting  - Request Routing  - Health Checks │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                        │
│  ┌─────────────┐  ┌─────────────┐    │   ┌─────────────┐  ┌─────────────┐     │
│  │    STT      │  │   LangGraph │◄───┴──►│     TTS     │  │   Memory    │     │
│  │   Engine    │  │    Agent    │        │   Engine    │  │   Manager   │     │
│  │             │  │   Runtime   │        │             │  │             │     │
│  │ - Whisper   │  │             │        │ - SoVITS    │  │ - Short-term│     │
│  │   (local)   │  │ - Intent    │        │   models    │  │ - Long-term │     │
│  │ - VAD       │  │   Routing   │        │ - Voice     │  │ - Entity    │     │
│  │   backup    │  │ - Tool Use  │        │   cloning   │  │   Extraction│     │
│  │             │  │ - Memory    │        │             │  │             │     │
│  └─────────────┘  └──────┬──────┘        └─────────────┘  └─────────────┘     │
│                          │                                                    │
│  ┌─────────────┐  ┌──────┴──────┐  ┌─────────────┐  ┌───────────────┐         │
│  │  LLM Host   │  │  Tool       │  │  Vector     │  │  Relational   │         │
│  │  (Ollama)   │  │  Registry   │  │  Store      │  │  Database     │         │
│  │             │  │             │  │             │  │               │         │
│  │ - llama3.1  │  │ - Home      │  │ - pgvector  │  │ - PostgreSQL  │         │
│  │ - phi4      │  │   Assistant │  │ - Sentence  │  │ - Conversa-   │         │
│  │ - qwen2.5   │  │ - Weather   │  │   Transform │  │   tions       │         │
│  │             │  │ - Calendar  │  │             │  │ - Device      │         │
│  │             │  │ - Custom    │  │             │  │   States      │         │
│  │             │  │   Skills    │  │             │  │ - Checkpoints │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────────┘         │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

1. **Privacy-First**: All voice processing, LLM inference, and data storage remain on local infrastructure
2. **Modularity**: Components are loosely coupled via defined APIs; any module can be replaced or upgraded independently
3. **Cost Efficiency**: Optimized for consumer hardware; targets break-even vs. cloud APIs within 6 months
4. **Resilience**: Graceful degradation when components fail; endpoints can cache commands during server downtime
5. **Extensibility**: LangGraph enables complex agent workflows without core system changes

---

## 2. Component Specifications

### 2.1 Endpoint Layer

#### Hardware Tiers

| Tier | Device | Capabilities | Cost Target | Use Case |
|------|--------|--------------|-------------|----------|
| **Basic** | ESP32-S3 + INMP441 | Wake word, raw PCM streaming | $12-18 | Simple rooms, background music control |
| **Standard** | Raspberry Pi Zero 2 W + ReSpeaker 2-Mic | Wake word, VAD, Opus encoding | $35-50 | Primary rooms, conversational use |
| **Advanced** | Raspberry Pi 4/5 + ReSpeaker 4-Mic | Edge STT (tiny), full VAD, local fallback | $80-120 | Kitchen, office, high-traffic areas |
| **Legacy** | Android Phone (Termux) | Full capabilities via app | $0 (repurpose) | Testing, temporary deployment |

#### Functional Requirements per Endpoint

- **Wake Word Detection**: Run locally using Porcupine (free tier) or openWakeWord
- **Voice Activity Detection (VAD)**: Silero VAD on Pi, simple energy-based VAD on ESP32
- **Audio Encoding**: Opus (preferred) for Pi tier, raw PCM for ESP32 tier
- **Network**: WiFi connectivity with MQTT for control, HTTP/WebSocket for audio
- **Indicators**: LED or display for listening/processing states
- **Power**: USB power or PoE where available

#### Communication Protocol

1. **Idle State**: Maintain MQTT connection, listen for wake word locally
2. **Activation**: On wake word detection, publish `voice/activate` to MQTT with room_id
3. **Streaming**: Open WebSocket or HTTP POST with Opus-encoded audio chunks
4. **Completion**: Server responds with audio file or stream; endpoint plays back

### 2.2 Server Layer

#### Hardware Specifications

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **CPU** | Intel i5-10400 / Ryzen 5 3600 | Intel i5-12400 / Ryzen 5 5600 | Intel i5-13400 / Ryzen 7 7700 |
| **GPU** | GTX 1060 6GB | RTX 3060 12GB | RTX 4060 Ti 16GB |
| **RAM** | 16GB DDR4 | 32GB DDR4/DDR5 | 64GB DDR5 |
| **Storage** | 256GB NVMe | 512GB NVMe + 2TB HDD | 1TB NVMe + 4TB HDD |
| **Network** | Gigabit Ethernet | 2.5GbE | 2.5GbE + WiFi 6 AP |

#### Service Architecture

**Core Services (Docker Compose Stack)**:

1. **PostgreSQL** (pgvector extension)
   - Conversational memory
   - Knowledge base storage
   - Device state tracking
   - LangGraph checkpoint persistence

2. **Ollama** (LLM Host)
   - Model management and serving
   - Primary: llama3.1:8b for general queries
   - Fallback: phi4 for faster responses
   - Optional: qwen2.5:14b for complex reasoning

3. **FastAPI Application** (Main API)
   - Voice endpoint handlers
   - STT/TTTS orchestration
   - LangGraph agent runtime
   - Authentication and session management

4. **Optional: Home Assistant** (Integration Bridge)
   - Device control abstraction
   - Automation engine
   - Third-party integrations

5. **Optional: MQTT Broker** (Mosquitto)
   - Device communication
   - Event broadcasting
   - Presence detection

### 2.3 Data Layer

#### PostgreSQL Schema Design

**Core Tables**:

1. **conversations**: Thread management, room context, metadata
2. **messages**: Message history with role, content, latency metrics, audio references
3. **knowledge**: RAG corpus with vector embeddings (pgvector)
4. **devices**: IoT device registry with state snapshots
5. **agent_checkpoints**: LangGraph state persistence for long-running workflows
6. **entities**: Extracted user preferences (names, preferences, facts)
7. **voice_profiles**: Speaker recognition data, voice cloning references

**Indexing Strategy**:

- GIN indexes on JSONB fields for flexible metadata queries
- IVFFlat index on vector embeddings for approximate nearest neighbor search
- Composite indexes on (room_id, updated_at) for conversation retrieval
- B-tree indexes on timestamps for time-series queries

---

## 3. Agent Architecture (LangGraph)

### 3.1 State Machine Design

The agent is implemented as a directed graph with conditional edges representing decision points.

#### State Object

```
AgentState:
├── messages: List[BaseMessage]          # Conversation history
├── room_id: str                         # Source room context
├── intent: Enum                         # Classified intent
├── context: Dict                        # Retrieved context
├── tool_calls: List[ToolCall]           # Pending tool executions
├── observations: List[str]              # Tool results
├── audio_response: Optional[bytes]      # Generated TTS output
├── latency_metrics: Dict                # Performance tracking
└── checkpoint_id: str                   # Persistence reference
```

#### Node Definitions

1. **intent_classifier**: Fast LLM call to route to appropriate handler
   - Outputs: QUERY, DEVICE, MEMORY, RAG, CLARIFICATION

2. **context_retriever**: Fetches relevant information
   - Recent conversation history (last 10 exchanges)
   - Room-specific device states
   - User entity preferences
   - Knowledge base (if RAG intent)

3. **device_handler**: Executes home automation
   - Parses natural language to device commands
   - Interfaces with Home Assistant or direct protocols
   - Returns execution confirmation

4. **memory_handler**: Manages long-term storage
   - Extracts and stores user preferences
   - Retrieves relevant past information
   - Summarizes old conversations

5. **rag_handler**: Knowledge-based responses
   - Vector search in knowledge base
   - Generates answers from retrieved documents
   - Falls back to general knowledge if no relevant docs

6. **general_handler**: Standard conversational responses
   - Uses LLM with system prompt and context
   - Optimized for concise, voice-friendly output

7. **clarification_handler**: Disambiguation
   - Asks follow-up questions when intent is unclear
   - Handles multi-turn clarification flows

8. **tts_generator**: Speech synthesis
   - Converts text response to audio
   - Selects appropriate voice based on room/user
   - Streams or returns complete audio file

### 3.2 Routing Logic

```
Entry → intent_classifier → context_retriever → [routing decision]

                                  ┌──► device_handler ──┐
                                  │                      │
              ┌──► DEVICE ────────┤                      │
              │                   │                      │
              ├──► MEMORY ───────► memory_handler ──────┤
              │                                          │
              ├──► RAG ─────────► rag_handler ──────────┤
              │                                          │
              ├──► CLARIFICATION ► clarification_handler │
              │                                          │
              └──► QUERY ───────► general_handler ──────┘
                                                           │
                                                           ▼
                                                    tts_generator → END
```

### 3.3 Tool System

**Built-in Tools**:

1. **home_control**: Light switches, thermostats, media players
2. **weather_fetch**: Local weather conditions and forecasts
3. **timer_alarm**: Set timers, alarms, reminders
4. **calendar_query**: Check schedules, add events
5. **music_control**: Play/pause/skip, playlist management
6. **search_knowledge**: Query internal knowledge base
7. **web_search**: DuckDuckGo integration for current information
8. **memory_store**: Save user preferences or facts
9. **memory_recall**: Retrieve stored information

**Tool Execution Flow**:

1. LLM generates tool call request
2. Tool registry validates and routes to implementation
3. Tool executes (may involve external API calls)
4. Observation returned to LLM
5. LLM generates final response based on observations

### 3.4 Memory Management

**Short-term Memory (Conversation Context)**:

- Sliding window of last 10 messages per conversation
- Automatic conversation segmentation (30-minute timeout)
- Context compression for very long conversations

**Long-term Memory (Entity Extraction)**:

- Automatic extraction of user preferences from conversations
- Storage in structured entity table
- Retrieval based on semantic similarity to current query

**Vector Memory (RAG)**:

- Documents ingested into knowledge base
- Embeddings generated using all-MiniLM-L6-v2 (384 dimensions)
- Similarity search with cosine distance threshold

---

## 4. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

**Goals**: Basic voice pipeline, single room proof-of-concept

**Deliverables**:
- PostgreSQL schema deployed
- FastAPI server with basic endpoints
- Whisper STT integration (faster-whisper small)
- SoVITS TTS integration (existing models)
- Single endpoint (Pi Zero) with wake word and streaming
- Simple intent routing (no tools yet)

**Success Criteria**:
- End-to-end latency < 5 seconds
- Basic Q&A functional
- Single conversation context maintained

### Phase 2: Agent Core (Weeks 3-4)

**Goals**: LangGraph integration, tool system

**Deliverables**:
- LangGraph workflow implementation
- Intent classification node
- Device control tool (Home Assistant integration)
- Basic RAG pipeline
- Conversation persistence

**Success Criteria**:
- Device commands execute correctly
- RAG answers questions from knowledge base
- Multi-turn conversations handled

### Phase 3: Distribution (Weeks 5-6)

**Goals**: Multiple endpoints, room awareness

**Deliverables**:
- Multi-endpoint deployment
- Room context injection
- Conflict resolution (multiple simultaneous activations)
- Endpoint tier support (ESP32 basic tier)

**Success Criteria**:
- 3+ rooms operational
- Room-specific context working ("turn on the lights" → correct room)
- Graceful handling of network issues

### Phase 4: Intelligence (Weeks 7-8)

**Goals**: Memory, personalization, advanced workflows

**Deliverables**:
- Entity extraction and storage
- User voice recognition
- Proactive suggestions based on patterns
- Complex multi-step workflows

**Success Criteria**:
- System remembers user preferences
- Voice profiles distinguish users
- Workflows like "good morning" trigger multiple actions

### Phase 5: Polish (Weeks 9-10)

**Goals**: Performance optimization, monitoring, edge cases

**Deliverables**:
- Latency optimization (< 3s target)
- Caching layer for frequent queries
- Monitoring and logging
- Error recovery and fallback modes

**Success Criteria**:
- 99% uptime
- < 3s average response time
- Graceful degradation when LLM offline

---

## 5. Technology Stack

### 5.1 Core Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Runtime** | Python 3.11+ | Ecosystem, ML library support |
| **API Framework** | FastAPI | Async native, OpenAPI generation, performance |
| **Agent Framework** | LangGraph | Stateful workflows, checkpointing, visualization |
| **LLM Interface** | LangChain + Ollama | Abstraction, local model support |
| **Database** | PostgreSQL 16 + pgvector | ACID compliance, vector search, JSONB flexibility |
| **STT** | faster-whisper | 4x speedup over OpenAI Whisper, local |
| **TTS** | SoVITS (existing) | High-quality voice cloning already implemented |
| **Wake Word** | Porcupine / openWakeWord | Efficient on-device detection |
| **VAD** | Silero VAD | Lightweight, accurate |
| **Audio Codec** | Opus | Low latency, good compression, voice-optimized |
| **Containerization** | Docker + Docker Compose | Deployment consistency, dependency isolation |
| **Process Management** | systemd (host) or Docker Swarm | Service reliability |

### 5.2 Development Tools

| Purpose | Tool |
|---------|------|
| **Testing** | pytest, pytest-asyncio, httpx |
| **Linting** | ruff, mypy |
| **Database Migrations** | alembic |
| **API Documentation** | FastAPI native OpenAPI |
| **Monitoring** | prometheus-client, grafana (optional) |
| **Logging** | structlog, loguru |

---

## 6. Data Flows

### 6.1 Voice Query Flow

```
1. USER SPEAKS → Endpoint mic array
2. WAKE WORD DETECTED → Local processing
3. VAD ACTIVATED → Recording begins
4. SILENCE DETECTED → Recording ends
5. OPUS ENCODING → Audio compressed
6. HTTP POST → Server /voice endpoint
7. OPUS DECODE → Server processes audio
8. WHISPER STT → Text extracted
9. INTENT CLASSIFICATION → Route determined
10. CONTEXT RETRIEVAL → DB queries executed
11. LLM GENERATION → Response generated
12. TOOL EXECUTION (if needed) → External APIs called
13. TTS GENERATION → SoVITS synthesizes audio
14. HTTP RESPONSE → Audio returned
15. PLAYBACK → Endpoint speaker output
```

**Target Latency Budget**:
- Steps 1-6: 100-500ms (depends on utterance length)
- Step 7: 50ms
- Step 8: 200-500ms (depends on model size)
- Steps 9-12: 500-2000ms (depends on complexity)
- Step 13: 500-2000ms (depends on text length)
- Steps 14-15: 100-300ms
- **Total Target: < 5s end-to-end**

### 6.2 Background Processing Flows

**Conversation Summarization**:
- Trigger: Conversation idle for 30 minutes
- Action: Summarize to key points, extract entities
- Storage: Update knowledge base with summary

**Knowledge Ingestion**:
- Trigger: User uploads document or URL
- Action: Chunk text, generate embeddings, store in pgvector
- Access: Available immediately for RAG queries

**Device State Sync**:
- Trigger: Periodic (every 5 seconds) or on change
- Action: Poll Home Assistant, update device table
- Benefit: Accurate context for device commands

---

## 7. Security Considerations

### 7.1 Network Security

- **API Authentication**: API keys or JWT tokens for endpoint authentication
- **TLS**: All communications over HTTPS/WSS
- **Network Segmentation**: IoT VLAN for endpoints, restrict inter-device communication
- **Firewall**: UFW or nftables on server, block unnecessary ports

### 7.2 Data Security

- **Encryption at Rest**: LUKS for database storage
- **PII Handling**: No cloud transmission of voice data or transcripts
- **Retention Policy**: Automatic deletion of audio files after processing (configurable)
- **Access Control**: Role-based access for multi-user homes

### 7.3 Privacy Features

- **Mute Button**: Physical mute on endpoints stops all processing
- **Activity LED**: Clear indication when listening/recording
- **Local Processing Guarantee**: No fallback to cloud APIs (configurable)
- **Audit Logging**: Track all device actions and data access

---

## 8. Scalability Planning

### 8.1 Vertical Scaling (Single Server)

| Bottleneck | Solution |
|------------|----------|
| **LLM Inference** | Upgrade GPU, use quantized models (int8, int4) |
| **Database** | Connection pooling, query optimization, read replicas |
| **STT Throughput** | Batch processing, model quantization |
| **Memory** | Increase RAM, add swap, use model offloading |

### 8.2 Horizontal Scaling (Future)

If server capacity exceeded:

1. **LLM Clustering**: vLLM with tensor parallelism across multiple GPUs
2. **Load Balancing**: Nginx upstream to multiple API instances
3. **Database Sharding**: Per-room partitioning if necessary
4. **Edge Compute**: Move STT to advanced endpoints (Pi 4 tier)

### 8.3 Current Limits

With recommended hardware (RTX 3060 12GB):
- **Concurrent STT streams**: 2-3 (Whisper small)
- **Concurrent LLM requests**: 1-2 (llama3.1:8b)
- **Simultaneous active conversations**: 5-8 rooms
- **Knowledge base size**: ~100k documents (pgvector limit)

---

## 9. Integration Points

### 9.1 Home Assistant Integration

**Approach**: WebSocket API connection

**Capabilities**:
- Device discovery and control
- Automation triggers
- Sensor data access
- Media player control

**Fallback**: Direct protocol support (MQTT, Zigbee2MQTT, ESPHome) if HA unavailable

### 9.2 Third-Party APIs (Optional)

| Service | Use Case | Privacy Level |
|---------|----------|---------------|
| **OpenWeatherMap** | Weather queries | Anonymous |
| **DuckDuckGo** | Web search | Anonymous |
| **CalDAV** | Calendar integration | Local server |
| **Spotify Connect** | Music control | OAuth, local control |

### 9.3 Mobile Interface (Future)

- Companion app for configuration
- Text-based chat interface
- Voice message playback
- Device management

---

## 10. Testing Strategy

### 10.1 Unit Testing

- Agent node logic
- Tool implementations
- Database queries
- Audio encoding/decoding

### 10.2 Integration Testing

- End-to-end voice pipeline
- Database transaction integrity
- LLM tool calling accuracy
- STT accuracy benchmarks

### 10.3 Performance Testing

- Latency measurement under load
- Concurrent request handling
- Memory usage profiling
- GPU utilization monitoring

### 10.4 User Acceptance Testing

- Command accuracy (intent recognition)
- Response quality (LLM evaluation)
- Voice clarity (TTS MOS scores)
- Multi-room coordination

---

## 11. Deployment Architecture

### 11.1 Directory Structure

```
/home/child-trafficking-department/PycharmProjects/OpenMOSS/
├── docker-compose.yml          # Main orchestration
├── .env                        # Environment variables
├── config/
│   ├── postgres/               # DB initialization scripts
│   ├── ollama/                 # Model download scripts
│   └── nginx/                  # Reverse proxy config (optional)
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py
│   │   ├── routes/
│   │   ├── middleware/
│   │   └── dependencies/
│   ├── agent/                  # LangGraph workflows
│   │   ├── graph.py
│   │   ├── nodes/
│   │   ├── tools/
│   │   └── prompts/
│   ├── stt/                    # Speech-to-text
│   │   └── whisper_engine.py
│   ├── tts/                    # Text-to-speech
│   │   └── sovits_engine.py
│   ├── database/               # Data layer
│   │   ├── models.py
│   │   ├── queries.py
│   │   └── migrations/
│   └── shared/                 # Utilities
│       ├── config.py
│       └── logging.py
├── models/                     # Local model storage
│   ├── whisper/
│   ├── sovits/
│   └── llm/
├── tests/
└── docs/
```

### 11.2 Environment Configuration

**Critical Environment Variables**:

```
# Database
DATABASE_URL=postgresql://user:pass@localhost/homeagent
DB_PASSWORD=secure_random_string

# LLM
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_LLM_MODEL=llama3.1:8b
FALLBACK_LLM_MODEL=phi4

# Security
API_KEY=secure_api_key_for_endpoints
JWT_SECRET=random_secret_for_tokens

# Audio
STT_MODEL_SIZE=small
TTS_DEFAULT_VOICE=default
AUDIO_CODEC=opus

# Integration
HOME_ASSISTANT_URL=http://homeassistant.local:8123
HA_TOKEN=long_lived_access_token

# Paths
MODEL_PATH=/app/models
AUDIO_CACHE_PATH=/tmp/audio_cache
```

---

## 12. Maintenance & Operations

### 12.1 Monitoring

**Metrics to Track**:
- End-to-end latency (p50, p95, p99)
- STT word error rate (periodic benchmarks)
- LLM tokens per second
- GPU utilization and temperature
- Database connection pool usage
- Endpoint online/offline status

**Logging**:
- Structured JSON logs for aggregation
- Log levels: ERROR, WARN, INFO, DEBUG
- Separate audio processing logs for debugging

### 12.2 Backup Strategy

- **Database**: Daily pg_dump to external storage
- **Models**: Version controlled, backup on change
- **Configuration**: Git-tracked, encrypted secrets
- **Knowledge Base**: Exportable to JSON

### 12.3 Update Procedures

- **Zero-downtime updates**: Blue-green deployment with Docker
- **Model updates**: Canary release with fallback
- **Database migrations**: Alembic with rollback scripts
- **Endpoint updates**: OTA for ESP32, apt for Pi

---

## 13. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| **GPU Failure** | High | CPU fallback (slower), cloud API fallback (configurable) |
| **Network Outage** | High | Local caching, endpoint offline mode with local commands |
| **STT Accuracy Poor** | Medium | Model fine-tuning on household voices, manual correction mode |
| **LLM Hallucination** | Medium | Constrained prompting, tool validation, user confirmation for critical actions |
| **Data Loss** | High | Automated backups, RAID storage, UPS for graceful shutdown |
| **Security Breach** | Medium | Network isolation, API authentication, no cloud exposure |

---

## 14. Future Roadmap

### Phase 6: Advanced Features (Months 3-6)

- **Multi-language support**: Automatic language detection, per-user language preferences
- **Voice Cloning**: Personal voice profiles for TTS
- **Computer Vision Integration**: Camera-based presence detection, gesture control
- **Predictive Actions**: ML-based prediction of user needs
- **Voice Prints**: Speaker identification for personalized responses

### Phase 7: Ecosystem (Months 6-12)

- **Mobile App**: iOS/Android companion
- **Third-party Skills**: Plugin system for community extensions
- **Cloud Bridge**: Optional hybrid mode for when away from home
- **Satellite Servers**: Multiple home locations synced
- **Voice Marketplace**: Pre-trained voice models, skill sharing

---

## 15. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **End-to-end Latency** | < 3 seconds | API timing headers |
| **Command Accuracy** | > 90% | Manual evaluation logs |
| **Uptime** | > 99% | Health check monitoring |
| **User Satisfaction** | > 4/5 | Periodic surveys |
| **Cost vs Cloud** | Break-even < 6 months | Usage tracking |
| **Privacy Score** | 100% local | Network monitoring |

---

## Appendix A: Hardware Shopping List

### Server Build (Recommended)

| Component | Specific Item | Est. Cost |
|-----------|---------------|-----------|
| GPU | RTX 3060 12GB (used) | $200 |
| CPU | Intel i5-12400F | $150 |
| Motherboard | B660 mATX | $100 |
| RAM | 32GB DDR4-3200 | $60 |
| Storage | 1TB NVMe SSD | $50 |
| PSU | 550W 80+ Bronze | $50 |
| Case | Mid-tower ATX | $50 |
| **Subtotal** | | **$660** |

### Endpoints (5-Room Setup)

| Item | Qty | Unit Cost | Total |
|------|-----|-----------|-------|
| Raspberry Pi Zero 2 W | 5 | $15 | $75 |
| ReSpeaker 2-Mic HAT | 5 | $20 | $100 |
| Micro-USB Power Supply | 5 | $5 | $25 |
| MicroSD Card 32GB | 5 | $8 | $40 |
| **Subtotal** | | | **$240** |

### Total Initial Investment: ~$900

---

## Appendix B: Glossary

- **VAD**: Voice Activity Detection - determines when speech starts/stops
- **STT**: Speech-to-Text - voice recognition (Whisper)
- **TTS**: Text-to-Speech - voice synthesis (SoVITS)
- **LLM**: Large Language Model - conversational AI (Llama, Phi)
- **RAG**: Retrieval-Augmented Generation - answering from documents
- **Opus**: Audio codec optimized for voice, low latency
- **LangGraph**: Framework for building stateful agent workflows
- **pgvector**: PostgreSQL extension for vector similarity search
- **MQTT**: Lightweight messaging protocol for IoT

---

*Document Version: 1.0*
*Last Updated: 2026-02-14*
*Status: Planning Phase*
