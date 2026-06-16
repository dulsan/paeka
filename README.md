# PAEKA

### Personal AI Engineering & Knowledge Assistant

PAEKA is a self-hosted AI assistant platform focused on engineering, research, software development, and knowledge management.

Unlike a simple chat interface, PAEKA combines local LLM inference, agentic retrieval, long-term memory, document intelligence, knowledge graphs, tool execution, and iterative reasoning into a unified system.

The project is designed for users who want a private, extensible AI assistant capable of working with large document collections, technical knowledge bases, research material, and engineering workflows.

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

Located in:

```text
backend/api/
```

Provides:

### Conversation Management

```text
/api/conversations
```

### Chat

```text
/api/conversations/{id}/chat
```

### Documents

```text
/api/documents
```

### Memory

```text
/api/memory
```

### Knowledge Graph

```text
/api/knowledge
```

### Models

```text
/api/models
```

### OpenAI Compatibility

```text
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

* Python 3.12+
* UV
* Docker Desktop
* Docker Compose

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
```

Development dependencies:

```bash
uv sync --extra dev
```

---

## 3. Start Weaviate

```bash
docker compose up -d
```

Verify:

```bash
curl http://localhost:8090/v1/.well-known/ready
```

---

## 4. Download a Model

Example:

```bash
uv run python scripts/download_model.py
```

Place GGUF models under:

```text
models/
```

---

## 5. Configure

Edit:

```text
config/settings.toml
```

Important sections:

```toml
[llm]
base_url = "http://localhost:8080/v1"

[retrieval]
enabled = true

[memory]
enabled = true

[knowledge_graph]
enabled = true
```

---

## 6. Start PAEKA

Windows:

```powershell
.\scripts\start.ps1
```

Manual:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

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

See:

```text
API_REFERENCE.md
```

for complete documentation.

---

# Development

Run tests:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check .
```

Format:

```bash
uv run ruff format .
```

Type checking:

```bash
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

Planned and partially implemented capabilities include:

* Enhanced multi-agent orchestration
* Improved tool execution
* Expanded knowledge graph reasoning
* SearXNG integration
* MCP integrations
* Advanced workflow automation
* Rich web frontend

---

# License

Licensed under the terms provided in the repository's `LICENSE` file.

---

# Disclaimer

PAEKA is an actively evolving project. Some documentation may describe architecture that is currently under migration or refactoring. Always use the codebase and `config/settings.toml` as the authoritative source of truth for deployment and configuration.

