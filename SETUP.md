# PAEKA — Setup & Development Guide (v0.7.0)

**Phases implemented:** 1 · 2 · 3 · 4 · 5 · 6 · Agentic RAG

---

## Prerequisites

| Tool | Minimum | Notes |
|---|---|---|
| Python | 3.12+ | 3.14.x fully supported |
| UV | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker + Compose | 24+ | Already installed |
| NVIDIA Container Toolkit | latest | Required for SGLang GPU |
| NVIDIA Driver | 525+ | Check: `nvidia-smi` |
| CUDA Toolkit | 13.2 | Matches `--index-url https://download.pytorch.org/whl/cu132` |
| VRAM | 20 GB+ | For Qwen3-14B-Instruct in bfloat16 |

---

## 1. Install UV

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 2. Install PyTorch for CUDA 13.2 first

> **Critical:** PyTorch must be installed from the CUDA wheel index before `uv sync`.
> The `[tool.uv.sources]` in `pyproject.toml` declares this index, so `uv sync`
> will use it automatically. However, if you have any issues, install manually:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Verify GPU is available after install:
```python
import torch
print(torch.cuda.is_available())   # must be True
print(torch.cuda.get_device_name(0))
```

---

## 3. Install Project Dependencies

```bash
cd paeka
uv sync
uv sync --extra dev    # adds pytest, ruff, pyright
```

`uv sync` will use the CUDA wheel index declared in `pyproject.toml` for torch/torchvision.

> **Note on Pillow:** `pyproject.toml` constrains `pillow>=12.0.0` which ships
> pre-built wheels for Python 3.14 on Windows. You should not see the `zlib`
> compile error that affected `pillow==10.4.0`.

> **Note on docling:** Removed in v0.7.0. Replaced by `pymupdf` which ships
> pre-built wheels for Python 3.14/Windows with no C compilation required.

---

## 4. Start Infrastructure (Docker)

```bash
# Start SGLang + Weaviate
docker compose up -d

# Watch startup — wait for both to be healthy before proceeding
docker compose logs -f
```

**SGLang** downloads ~28 GB of model weights on first run. Subsequent starts are fast.

Verify both services are up:
```bash
curl http://localhost:30000/health        # → {"status":"healthy"}
curl http://localhost:8080/v1/.well-known/ready  # → {}
```

---

## 5. Configure

All configuration is in `config/settings.toml`.

Key settings:
```toml
[llm]
base_url = "http://localhost:30000/v1"

[retrieval]
enabled      = true
embed_device = "cuda"   # "cpu" if running embeddings on CPU
max_hops     = 2        # agentic RAG loop iterations

[knowledge_graph]
enabled = true

[memory]
enabled = true
```

Environment variable overrides (no restart needed):
```bash
export PAEKA_LLM__BASE_URL=http://192.168.1.5:30000/v1
export PAEKA_RETRIEVAL__EMBED_DEVICE=cpu
export PAEKA_RETRIEVAL__MAX_HOPS=3
```

---

## 6. Start the API Server

```bash
# Development (hot-reload)
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## 7. Verify

```bash
curl http://localhost:8000/api/health
```

Expected:
```json
{
  "status": "ok",
  "version": "0.7.0",
  "components": {
    "llm": true,
    "retrieval": true,
    "memory": true,
    "knowledge_graph": true,
    "skills": true
  }
}
```

---

## 8. Run Tests

```bash
uv run pytest tests/ -v
```

All unit tests run without GPU, Weaviate, or SGLang. Integration tests use
in-memory SQLite and mocked LLM.

---

## 9. Lint & Type Check

```bash
uv run ruff check . --fix
uv run ruff format .
uv run pyright
```

---

## Agentic RAG Pipeline

When retrieval is enabled, every chat message flows through:

```
User query
  └─ Planner      : decomposes into 2-6 sub-queries with tool assignments
  └─ Retriever    : executes sub-queries (vector / graph / keyword)
  └─ Critic       : evaluates sufficiency, prunes bad passages
       └─ loop back to Retriever if insufficient (max_hops iterations)
  └─ Synthesiser  : composes final answer from approved passages
```

SSE events from the chat endpoint:
```
data: {"type": "plan",    "content": "Search for X then check KG..."}
data: {"type": "context", "sources": [...], "graph": "...", "hops": 2}
data: {"type": "delta",   "content": "The transformer..."}
data: {"type": "done",    "message_id": "uuid"}
```

---

## Skills System

Skills are DIRECTORIES in `backend/skills/definitions/`, each containing a `SKILL.md` file.

Structure per skill:
```
my_skill/
  SKILL.md          ← required: TOML frontmatter + system prompt instructions
  config.toml       ← optional: overrides frontmatter values
  scripts/          ← optional: helper scripts
  references/       ← optional: API docs, examples
  assets/           ← optional: templates, data
```

`SKILL.md` frontmatter (TOML, fenced by `---`):
```toml
name        = "my_skill"
description = "Use when the task involves X. Written for model routing."
tags        = ["domain"]
temperature = 0.4
```

The description field is written FOR the model (routing/trigger text), not humans —
following the Claude Code skills pattern.

Hot-reload after editing: `POST /api/skills/reload`

---

## Document Ingestion

```bash
# Upload a PDF
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@paper.pdf"

# Extract knowledge graph from it
curl -X POST http://localhost:8000/api/knowledge/extract/<document_id>

# Run graph refinement passes
curl -X POST http://localhost:8000/api/knowledge/refine
```

Document parser selection:
| File | Parser |
|---|---|
| `.pdf` / `.docx` / `.html` | PyMuPDF (primary) → plaintext fallback |
| `.md` / `.markdown` | Marker structural parser |
| `.xlsx` / `.csv` | openpyxl + pandas |
| `.txt` / `.rst` | plaintext paragraph splitter |

---

## Project Layout (v0.7.0)

```
paeka/
├── backend/
│   ├── agent/                      ← Agentic RAG (LangGraph)
│   │   ├── graph.py                ← pipeline: Planner→Retriever→Critic→Synthesiser
│   │   ├── state.py                ← AgentState TypedDict
│   │   └── nodes/
│   │       ├── planner.py          ← query decomposition
│   │       ├── retriever.py        ← multi-tool multi-hop retrieval
│   │       ├── critic.py           ← sufficiency evaluation + loop control
│   │       └── synthesiser.py      ← final answer generation
│   │
│   ├── api/routes/
│   │   ├── health.py               ← component readiness
│   │   ├── conversations.py        ← conversation CRUD
│   │   ├── chat.py                 ← agentic SSE chat
│   │   ├── documents.py            ← ingest / manage / refresh
│   │   ├── memory.py               ← inspect / reset memory
│   │   ├── knowledge.py            ← KG extract / refine / query
│   │   └── skills.py               ← list / get / reload
│   │
│   ├── llm/
│   │   └── client.py               ← pure httpx (no openai/meta SDK)
│   │
│   ├── retrieval/
│   │   ├── embedder.py             ← BGE-M3
│   │   ├── reranker.py             ← BGE-Reranker-Large
│   │   ├── weaviate_store.py       ← hybrid search
│   │   ├── chunker.py              ← layout-aware chunker
│   │   └── engine.py               ← embed→search→rerank (alpha override)
│   │
│   ├── ingestion/
│   │   ├── pipeline.py             ← versioning + semantic dedup
│   │   ├── repository.py           ← document/chunk CRUD
│   │   └── parsers/
│   │       ├── base.py             ← ParsedDocument, ElementType
│   │       ├── dispatcher.py       ← auto-selects parser by file type
│   │       ├── pymupdf_parser.py   ← primary PDF/DOCX (Py3.14 safe)
│   │       ├── marker_parser.py    ← Markdown structural parser
│   │       └── spreadsheet_parser.py
│   │
│   ├── memory/
│   │   ├── repository.py           ← conversation + message CRUD
│   │   └── service.py              ← session window + summarisation + global extraction
│   │
│   ├── knowledge/
│   │   ├── graph.py                ← KGNode/KGEdge + repository
│   │   ├── extractor.py            ← LLM entity/relation extraction
│   │   ├── refinement.py           ← merge/prune/validate passes
│   │   └── retriever.py            ← graph-aware context augmentation
│   │
│   ├── skills/
│   │   ├── manager.py              ← folder-based skill loader
│   │   └── definitions/
│   │       ├── engineering_analysis/SKILL.md
│   │       ├── scientific_research/SKILL.md
│   │       ├── software_architecture/SKILL.md
│   │       └── documentation_review/SKILL.md
│   │
│   └── shared/
│       ├── config.py               ← Pydantic settings
│       ├── database.py             ← async SQLite
│       └── logging.py              ← Rich logging
│
├── tests/
│   ├── unit/
│   │   ├── test_agent_nodes.py     ← planner/retriever/critic/synthesiser
│   │   ├── test_llm_client.py      ← httpx mock transport tests
│   │   ├── test_chunker.py
│   │   ├── test_parsers.py
│   │   ├── test_deduplication.py
│   │   ├── test_knowledge_graph.py
│   │   └── test_skills.py
│   └── integration/
│       └── test_api.py
│
├── config/settings.toml
├── docker-compose.yml              ← SGLang + Weaviate
├── API_REFERENCE.md
├── SETUP.md
├── main.py
└── pyproject.toml
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pillow` build failure (`zlib`) | Should not occur with `pillow>=12.0.0`. Run `uv sync --upgrade` |
| `torch` not finding CUDA | `pip install torch --index-url https://download.pytorch.org/whl/cu132` |
| `llm: false` in health | SGLang still loading. `docker compose logs sglang` |
| `retrieval: false` | `[retrieval] enabled = true` in settings.toml |
| Weaviate refused | `docker compose up weaviate -d` |
| Out of VRAM | Reduce `--mem-fraction-static` in docker-compose.yml |
| `ModuleNotFoundError: backend` | Run from project root with `uv run` |
| `fitz` not found | `uv add pymupdf` |

---

## Downloading Your Model

```bash
# Download the recommended model (Qwen3.6 35B Q4_K_M for 8GB VRAM + 32GB RAM)
uv run python scripts/download_model.py \
    --repo  unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
    --file  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --dest  models/qwen

# List models you already have
uv run python scripts/download_model.py --list

# After download, restart the inference container
docker compose restart paeka-llamacpp
```

The model lands at `models/qwen/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
The path is already the default in `.env.example` and `settings.toml`.

---

## GPU Tuning (8GB VRAM + 32GB RAM)

Set in `.env`:

```bash
LLAMA_GPU_LAYERS=35     # layers on GPU; increase until CUDA OOM, then reduce by 5
LLAMA_CTX_SIZE=8192     # context window; reduce to 4096 if VRAM is tight
LLAMA_N_BATCH=512       # prompt processing batch size
LLAMA_CHAT_TEMPLATE=chatml   # Qwen uses ChatML
```

Check VRAM usage after startup:
```bash
nvidia-smi
```

If you see CUDA OOM in `docker compose logs paeka-llamacpp`:
```bash
# Reduce GPU layers by 5 and restart
LLAMA_GPU_LAYERS=30 docker compose restart paeka-llamacpp
```

---

## New Features (v0.11.0)

### Autonomous Iteration
```bash
curl -X POST http://localhost/api/agent/iterate \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Write a Python implementation of binary search with full type hints",
    "max_iterations": 4,
    "score_threshold": 0.85,
    "skill": "software_architecture"
  }'
```

### Self-Healing Tool Calling
```bash
curl -X POST http://localhost/api/agent/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "request": "Search for recent work on MoE architectures and summarise findings",
    "tools": ["web_search", "retrieve"],
    "max_retries": 3
  }'
```

### Sandboxed Code Execution
```bash
curl -X POST http://localhost/api/sandbox/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "import math; print(math.pi)", "language": "python"}'
```

---

## Document Parsers (v0.11.0)

| File type | Parser | Notes |
|---|---|---|
| `.pdf` / `.docx` / `.pptx` / `.html` | Docling 2.9+ + HybridChunker | Multi-modal, layout-aware |
| `.tex` / `.latex` | LaTeX structural parser | No compilation needed |
| `.md` / `.markdown` | Marker structural parser | |
| `.xlsx` / `.csv` | openpyxl + pandas | |
| `.txt` / `.rst` | Plaintext | |
| `.py` / `.ts` / `.c` / `.cpp` | Tree-sitter (optional) | `uv sync --extra code` |

PyMuPDF has been removed. Docling 2.9.0+ natively supports Python 3.14 and
handles multi-modal parsing (text, tables, equations, figures) significantly
better than PyMuPDF for complex documents.
