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

* Runs against local LLM backends (Ollama by default; llama.cpp, LiteLLM, and SGLang providers also available)
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
* Extensible tool architecture

### Native Windows Support

* PowerShell launch scripts
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
│ Memory   │ │ Qdrant   │ │ KG Store │
└──────────┘ └──────────┘ └──────────┘
      |           |           |
      ▼           ▼           ▼
┌─────────────────────────────────────┐
│         Local LLM Backend           │
│      Ollama (default) / llama.cpp   │
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
* Qdrant integration
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

├── docs/

├── models/

├── scripts/
│
└── tests/
```
---

# Requirements

## Software

* Python 3.12+ (preferably v3.14.2)
* UV Package Manager

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

## 2. Install

PAEKA runs natively on the host, no containers.

```bash
uv sync --extra dev
```

Start Qdrant:

```bash
# Download qdrant.exe (or the Linux/macOS equivalent) from:
# https://github.com/qdrant/qdrant/releases
.\bin\qdrant.exe --config-path config\qdrant.yaml
```

Start Ollama and import the model:

```bash
ollama serve
# in another terminal:
ollama create paeka-qwen -f models\qwen\Modelfile
```

Configure (optional) — edit `config/settings.toml`, or override via `.env`
(copy `.env.example` to `.env` first):

```toml
[llm]
provider = "ollama"
base_url = "http://localhost:11434/v1"
model = "paeka-qwen"

[retrieval]
enabled = true
qdrant_url = "http://localhost:6333"
```

Start PAEKA:

```bash
uv run python main.py
```

**Note:** if your checkout lives inside OneDrive (or another sync
client), exclude the folder from sync — OneDrive's placeholder/sync layer
has caused real data-loss and slow-file-access issues with Qdrant's
storage and the multi-GB GGUF in past runs.

The app is available at `http://localhost:8000`. API docs (if
enabled in settings) at `http://localhost:8000/docs`.

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

For complete documentation see the following document:

```text
API_REFERENCE.md
```
---

# Development

```bash
# First time / after pulling changes:
uv sync --extra dev

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
| API_REFERENCE.md                      | REST API documentation   |
| FIXES.md                              | Known issues and fixes   |
| docs/REVIEW.md                        | Code review findings     |
| ~~SETUP.md~~                          | **Outdated** -- written for a previous SGLang + Weaviate architecture (tagged v0.7.0; current is v0.11.3+). Use this README instead until it's rewritten. |

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

## Phase 4: Advanced Architecture

- **Agentic Orchestration:** `deepagents` harness on top of LangGraph for planning, sub-agent delegation, and human-in-the-loop approval gates on tool calls.
- **Agentic Graph RAG:** FalkorDB (via `falkordblite`, embedded, no server) as a Cypher query layer over the existing SQLite knowledge graph.
- **Production Hardening:** Structured error handling, graceful degradation, and comprehensive observability (structlog).

---

# License

Licensed under the terms provided in the repository's `LICENSE` file.

---

# Disclaimer

PAEKA is an actively evolving project. Some documentation may describe architecture that is currently under migration or refactoring. Always use the codebase and `config/settings.toml` as the authoritative source of truth for deployment and configuration.

