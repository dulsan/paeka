# PAEKA

### Personal AI Engineering & Knowledge Assistant

PAEKA is a self-hosted AI assistant platform focused on engineering, research, software development, and knowledge management.

Unlike a simple chat interface, PAEKA combines local LLM inference, agentic retrieval, long-term memory, document intelligence, knowledge graphs, tool execution, and iterative reasoning into a unified system.

The project is designed for users who want a private, extensible AI assistant capable of working with large document collections, technical knowledge bases, research material, and engineering workflows.

---

## ⚠️ Status: Unstable & Experimental

**PAEKA is currently in active development and should be used with caution.** The system is undergoing significant architectural changes, and breaking changes may occur without notice. Use in production environments at your own risk. Features are subject to change, and data loss or unexpected behavior may occur.

---

## Features

### Local-First AI Assistant

* Runs against local `llama.cpp` servers
* OpenAI-compatible API support
* No cloud dependency required
* Configurable local GGUF models

### Agentic RAG

* Multi-stage retrieval pipeline
* Query decomposition and planning
* Hybrid retrieval
* Re-ranking
* Multi-hop retrieval
* Retrieval critique and refinement loops

### Knowledge Graph

* Automatic entity extraction
* Relationship discovery
* Graph-assisted retrieval
* Structured knowledge exploration

### Long-Term Memory

* Conversation history persistence
* Automatic summarization
* Session memory
* Global memory management

### Document Intelligence

* PDF ingestion
* DOCX ingestion
* XLSX ingestion
* Markdown support
* Plain-text ingestion
* Versioned document management

### Skills Framework

* Domain-specific assistant behaviors
* Engineering workflows
* Research workflows
* Custom skill definitions

### OpenAI-Compatible API

* `/v1/models`
* `/v1/chat/completions`
* Streaming support
* Compatible with third-party OpenAI clients

### Tool Calling

* Web search integration
* Sandboxed execution
* Extensible tool architecture

### Native Windows Support

* PowerShell launch scripts
* Dockerized infrastructure
* Local llama.cpp deployment

---

# Architecture

```text
┌─────────────────────────────────────┐
│               Client                │
│  Web UI / API / OpenAI Clients      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│            FastAPI API              │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│           Agent Pipeline            │
├─────────────────────────────────────┤
│ Planner                             │
│ Retriever                           │
│ Critic                              │
│ Synthesiser                         │
└─────────────────┬───────────────────┘
                 |
      ┌───────────---───────────┐
      ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Memory   │ │ Weaviate │ │ KG Store │
└──────────┘ └──────────┘ └──────────┘
      |           |           |
      ▼           ▼           ▼
┌─────────────────────────────────────┐
│         Local LLM Backend           │
│         llama.cpp Server            │
└─────────────────────────────────────┘
```

---

# Core Components

## Agent System

Located in:

```text
backend/agent/
```

Contains:

* Planner
* Retriever
* Critic
* Synthesiser
* Iterative reasoning graphs
* Tool execution graphs
* ReAct-style workflows

Key files:

```text
backend/agent/graph.py
backend/agent/iteration_graph.py
backend/agent/react_graph.py
backend/agent/tool_graph.py
```

---

## API Layer

```text
# Located in:
backend/api/

# Provides:
# Conversation Management
/api/conversations

# Chat
/api/conversations/{id}/chat

# Documents
/api/documents

# Memory
/api/memory

# Knowledge Graph
/api/knowledge

# Models
/api/models

# OpenAI Compatibility
/v1/models
/v1/chat/completions
```
---

## Retrieval System

Located in:

```text
backend/retrieval/
```

Capabilities:

* Vector search
* Hybrid search
* Re-ranking
* Multi-hop retrieval
* Weaviate integration
* Embedding generation

Default models:

```text
BAAI/bge-m3
BAAI/bge-reranker-large
```

---

## Knowledge Graph

Located in:

```text
backend/knowledge/
```

Provides:
* Entity extraction
* Relationship extraction
* Graph storage
* Graph-assisted retrieval

Supported entity types include:
* Concepts
* Algorithms
* Frameworks
* Papers
* Datasets
* Tools
* Organizations
* People

---

## Memory System

Located in:

```text
backend/memory/
```

Features:
* Conversation persistence
* Session memory
* Automatic summarization
* Memory limits
* Long-term recall

---

## Document Ingestion

Located in:

```text
backend/ingestion/
```

Supported formats:

| Format   | Supported |
| -------- | --------- |
| PDF      | ✓         |
| DOCX     | ✓         |
| XLSX     | ✓         |
| TXT      | ✓         |
| Markdown | ✓         |
| HTML     | ✓         |
| LaTeX    | ✓         |

Features:
* Chunking
* Deduplication
* Versioning
* Metadata extraction

---

# Repository Structure

```text
paeka/
│
├── backend/
│   ├── agent/
│   ├── api/
│   ├── ingestion/
│   ├── knowledge/
│   ├── llm/
│   ├── memory/
│   ├── retrieval/
│   ├── security/
│   ├── skills/
│   └── tools/
│
├── config/
│   └── settings.toml
│
├── database/
│
├── docs/
│
├── infra/
│   ├── caddy/
│   └── searxng/
│
├── models/
│
├── scripts/
│
└── tests/
```
---

# Requirements

## Software

* Python 3.12+ (preferably v3.14.2)
* UV Package Manager
* Docker Desktop 

## Optional

* NVIDIA GPU
* CUDA-capable drivers

---

# Installation

## 1. Clone

```bash
git clone https://github.com/dulsan/paeka.git
cd paeka
```

## 2. Install Dependencies

```bash
uv sync
# Development dependencies:
uv sync --extra dev
```

---

## 3. Start Qdrant Vector Store

```bash
# Download qdrant.exe from:
# https://github.com/qdrant/qdrant/releases
# Look for: qdrant-x86_64-pc-windows-msvc.zip

# Run in a terminal:
.\bin\qdrant.exe

# Verify:
curl http://localhost:6333/healthz
```

---

## 4. Start Ollama (LLM Backend)

```bash
# Download from: https://ollama.ai
# Then start the service:
ollama serve
```

In another terminal, create the model:

```bash
ollama create paeka-qwen -f models\qwen\Modelfile
```

Verify:

```bash
curl http://localhost:11434/v1/models
```

---



---

## 5. Configure (Optional)

Edit:

```text
config/settings.toml
```

Important sections:

```toml
[llm]
provider = "ollama"
base_url = "http://localhost:11434/v1"
model = "paeka-qwen"

[retrieval]
enabled = true

[memory]
enabled = true

[tools]
web_search_enabled = false  # Set true to enable with SearXNG

[sandbox]
enabled = true  # Requires Docker Desktop
```

---

## 6. Start PAEKA

Windows (PowerShell):

```powershell
cd "C:\path\to\paeka"
uv sync
uv run python main.py
```

**Note:** OneDrive integration can cause synchronization issues. If you experience problems, ensure the paeka directory is excluded from OneDrive sync.

Manual with uv:

```bash
uv run python main.py
```

Or direct uvicorn:

```bash
uv run uvicorn backend.api.app:create_app --host 0.0.0.0 --port 8000 --factory
```

The application will be available at: `http://localhost:8000`

API documentation: `http://localhost:8000/docs` (if enabled in settings)

---

# API Endpoints

| Endpoint                          | Description            |
| --------------------------------- | ---------------------- |
| GET /api/health                   | Health status          |
| GET /api/conversations            | List conversations     |
| POST /api/conversations           | Create conversation    |
| POST /api/conversations/{id}/chat | Chat                   |
| POST /api/documents/upload        | Upload document        |
| GET /api/documents                | List documents         |
| GET /api/memory                   | Memory API             |
| GET /api/models                   | Available models       |
| POST /v1/chat/completions         | OpenAI-compatible chat |

For complete documentation see the follwing document:

```text
API_REFERENCE.md
```
---

# Development

```bash
# Run tests:
uv run pytest

Lint:
uv run ruff check .

Format:
uv run ruff format .

Type checking:
uv run pyright
```
---

# Documentation

| File                                  | Purpose                  |
| ------------------------------------- | ------------------------ |
| SETUP.md                              | Installation and setup   |
| API_REFERENCE.md                      | REST API documentation   |
| FIXES.md                              | Known issues and fixes   |
| docs/REVIEW.md                        | Code review findings     |
| WEAVIATE_RAFT_CLUSTER_ISSUE_REPORT.md | Weaviate troubleshooting |

---

# Roadmap

PAEKA development follows a phased approach focused on defensive, observable development with measurable stability milestones.

## Phase 1: Establish Observability & Safety Rails (Current)

The goal: **"Never debug blind."** Build diagnostic frameworks and execution guardrails so every failure produces a clean signal.

- **Local Logfire Tracing:** Structured tracing of LLM prompts, completions, tool calls, and latency across LangGraph, langchain-ollama, and HTTPX.
- **Orchestration Guardrails:** Call-memoization and circuit-breaker logic to prevent tool looping and consecutive failures. Failures are logged as deliberate events in the trace.
- **ReAct Loop Validation:** End-to-end tool calling with native LLM function calling (langchain-ollama ChatOllama) to verify schema translation and MCP tool dispatch.
- **MCP Tool Discovery:** In-process tool manager fallback to avoid transport round-trips during initialization. (✅ Completed)

## Phase 2: Core Parity & State Verification

With a visible, stable loop, wire up missing data layers confidently.

- **GraphRAG via MCP:** Build the `graph_search` tool for entity-relation queries.
- **Conversation Memory on Qdrant:** Validate history summarization and semantic retrieval against the vector store.
- **Tool Call Tracing:** Deep visibility into which tools are called, their arguments, and results in the trace.

## Phase 3: Environment Polish (Pre-Docker Stabilization)

Use hard-learned lessons to build a defensive local environment.

- **Interactive Pre-Flight Wizard:** Diagnostics for OneDrive blocks, port conflicts, unsigned binaries, and environment validation.
- **Second-Round Stabilization Testing:** End-to-end system verification using the wizard to simulate a clean install.
- **Windows Integration Hardening:** Resolve OneDrive synchronization issues and file lock interactions.

## Phase 4: Dockerization & Advanced Features

- **Containerization:** Transition to `docker-compose` for reproducible, isolated deployments. Self-hosted Langfuse integration.
- **Advanced Architecture:** Hardened code sandboxes, long-context graph compression, CPU-bound router model testing.
- **Production Hardening:** Structured error handling, graceful degradation, and comprehensive observability.

---

# License

Licensed under the terms provided in the repository's `LICENSE` file.

---

# Disclaimer

PAEKA is an actively evolving project. Some documentation may describe architecture that is currently under migration or refactoring. Always use the codebase and `config/settings.toml` as the authoritative source of truth for deployment and configuration.

