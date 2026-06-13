# PAEKA API Reference (v0.7.0)

Base URL: `http://localhost:8000/api`

All request/response bodies are JSON. Streaming endpoints use `text/event-stream`.
Interactive Swagger UI is intentionally disabled — use this document.

---

## Health

### `GET /api/health`

**Response** `200`
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

All components default to `false` until their respective config flags are enabled.

---

## Conversations

### `GET /api/conversations`
List all conversations, newest first.

### `POST /api/conversations`
```json
{ "title": "New Conversation" }
```
**Response** `201` — conversation object.

### `GET /api/conversations/{id}`
Returns conversation with full message history.

### `PATCH /api/conversations/{id}`
```json
{ "title": "Renamed" }
```

### `DELETE /api/conversations/{id}`
**Response** `204`

---

## Chat

### `POST /api/conversations/{id}/chat`

Send a message and receive a streaming response via Server-Sent Events.

**Request**
```json
{
  "message": "Explain the attention mechanism in transformers.",
  "skill": "scientific_research"
}
```

- `skill` (optional) — name of a skill to activate for this turn.

**Response** `text/event-stream`

Events arrive in this order:

```
data: {"type": "plan", "content": "Search for attention papers, then check KG."}

data: {"type": "context", "sources": [...], "graph": "...", "hops": 1}

data: {"type": "delta", "content": "The attention mechanism"}
data: {"type": "delta", "content": " allows the model to..."}

data: {"type": "done", "message_id": "uuid"}
```

**`plan` event** — emitted after the Planner node runs (only when retrieval is enabled).
```json
{
  "type": "plan",
  "content": "Search semantic index for attention mechanism, check KG for Transformer entity."
}
```

**`context` event** — emitted before the first token, after Critic approves passages.
```json
{
  "type": "context",
  "sources": [
    {
      "filename": "attention_is_all_you_need.pdf",
      "heading": "3.2 Scaled Dot-Product Attention",
      "page": 4,
      "score": 0.934,
      "element_type": "text"
    }
  ],
  "graph": "<knowledge_graph>\n• Transformer --[USES]--> Attention\n</knowledge_graph>",
  "hops": 1
}
```
`hops` is the number of Critic→Retriever loop iterations that ran.

**`delta` event** — one text chunk.

**`done` event** — stream complete.

**`error` event** — exception occurred; stream closes.
```json
{ "type": "error", "detail": "SGLang endpoint unreachable" }
```

When retrieval is **disabled**, the `plan` and `context` events are skipped
and the assistant streams tokens directly.

---

## Documents

### `POST /api/documents/upload`
Multipart file upload. Accepts PDF, DOCX, XLSX, CSV, TXT, MD.

**Response** `202`
```json
{ "document_id": "uuid", "status": "processing" }
```

Re-uploading an unchanged file (same SHA-256) is idempotent — returns immediately.
Re-uploading a changed file retires the old version and re-ingests.

**Errors:** `415` unsupported type · `413` file too large · `503` retrieval disabled.

### `POST /api/documents/ingest-text`
```json
{ "text": "Document text...", "filename": "notes.txt" }
```
**Response** `202` — same shape as upload.

### `GET /api/documents`
List all documents with status, chunk count, parser used, and version number.

Document status values: `pending` → `processing` → `ready` | `failed`

### `GET /api/documents/{id}`
Returns document with all chunk metadata.

`element_type` values per chunk: `text` · `table` · `code` · `equation` · `figure` · `list_item` · `caption`

### `DELETE /api/documents/{id}`
Permanently deletes from Weaviate + SQLite. **Irreversible.**

### `POST /api/documents/{id}/refresh`
Force re-ingest from original file path. Retires current version.

### `GET /api/documents/{id}/chunks`
List all chunks (without full document metadata).

---

## Memory

### `GET /api/memory/global`
List all long-term facts extracted from conversations.

### `DELETE /api/memory/global`
Clear all global memories.

### `GET /api/memory/session/{conversation_id}`
Get the current session summary (may be `null`).

### `DELETE /api/memory/session/{conversation_id}`
Clear the session summary for a conversation.

---

## Knowledge Graph

### `GET /api/knowledge/stats`
```json
{ "nodes": 312, "edges": 847, "types": {"Algorithm": 45, "Concept": 120} }
```

### `POST /api/knowledge/extract/{document_id}`
Run KG extraction over a document's chunks. Returns entity/relation counts.

Extraction is additive — re-running merges with existing nodes.

### `POST /api/knowledge/refine`
Run all configured refinement passes.
```json
{ "passes": { "merge_duplicates": 7, "prune_weak_edges": 23, "validate_types": 4 } }
```

### `GET /api/knowledge/nodes`
Query params: `entity_type` (filter), `min_confidence` (float, default 0.0).

### `GET /api/knowledge/nodes/{id}`
Returns node with one-hop outgoing and incoming edges.

### `DELETE /api/knowledge/nodes/{id}`
Deletes node and all its edges.

### `GET /api/knowledge/query?q={text}`
Returns graph context matching a query. Useful for inspecting what the
graph knows before asking a question.

---

## Skills

### `GET /api/skills`
List all skills with name, description, tags, temperature, and system_prompt.

Built-in skills: `engineering_analysis` · `scientific_research` · `software_architecture` · `documentation_review`

### `GET /api/skills/{name}`
Get one skill.

### `POST /api/skills/reload`
Hot-reload skill folders from disk. Returns `{"loaded": N}`.

---

## Adding a Custom Skill

1. Create `backend/skills/definitions/my_skill/SKILL.md`:

```markdown
---
name        = "my_skill"
description = "Use when the task involves <X>. Written for model routing."
tags        = ["custom"]
temperature = 0.4
---

## My Skill Mode

<system prompt instructions here>

### Approach
...

### Gotchas
...
```

2. `POST /api/skills/reload` — no restart needed.

3. Use in chat: `{ "message": "...", "skill": "my_skill" }`

The `description` field should be written FOR the model (trigger/routing text)
— following the Anthropic Claude Code skills pattern where descriptions tell
the model when to apply the skill, not when humans should select it.

---

## Error Responses

```json
{ "detail": "Human-readable error message" }
```

| Code | Meaning |
|---|---|
| 400 | Bad request |
| 404 | Resource not found |
| 413 | File too large (>100 MB) |
| 415 | Unsupported media type |
| 503 | Component not enabled or not ready |
| 500 | Internal server error |

---

## Sandbox

### `GET /api/sandbox/status`

Check whether Docker is available for sandbox execution.

```json
{ "docker_available": true, "supported_languages": ["python", "bash", "javascript"] }
```

### `GET /api/sandbox/languages`

List supported execution languages.

### `POST /api/sandbox/execute`

Execute code in an isolated Docker container.

Security: `--network=none`, `--read-only`, `--cap-drop=ALL`, `--memory=256m`, `--pids-limit=64`

**Request**
```json
{
  "code": "print('Hello, PAEKA!')",
  "language": "python",
  "timeout": 30
}
```

**Response** `200`
```json
{
  "stdout": "Hello, PAEKA!\n",
  "stderr": "",
  "exit_code": 0,
  "timed_out": false,
  "success": true,
  "output": "Hello, PAEKA!"
}
```

**Errors**
- `400` — unsupported language or code blocked by content scanner.
- `503` — Docker not available.

---

## Agent Features

### `POST /api/agent/iterate`

Run the autonomous iteration loop (Generate → Evaluate → Reflect → repeat).

**Request**
```json
{
  "task": "Write a Python function that efficiently finds prime numbers up to N.",
  "context": "The function will be called millions of times per second.",
  "max_iterations": 4,
  "score_threshold": 0.85,
  "skill": "software_architecture"
}
```

**Response** `200`
```json
{
  "final_output": "def sieve_of_eratosthenes(n): ...",
  "iterations": 3,
  "final_score": 0.91,
  "converged": true,
  "critique_history": [
    "[Iteration 1] The initial implementation uses trial division which is O(n√n)...",
    "[Iteration 2] The sieve implementation is correct but lacks bounds checking..."
  ]
}
```

The loop exits when `final_score >= score_threshold` or `iterations >= max_iterations`.

### `POST /api/agent/tools/execute`

Run the self-healing tool calling pipeline.

**Request**
```json
{
  "request": "Search for recent papers on mixture-of-experts models and format the results",
  "tools": ["web_search", "retrieve", "format_code"],
  "max_retries": 3
}
```

**Response** `200`
```json
{
  "final_response": "Here are the most relevant recent papers on MoE...",
  "succeeded": true,
  "iterations": 2,
  "reflections": [],
  "results": [
    {
      "tool": "web_search",
      "success": true,
      "output": "[Mixtral 8x7B] ...",
      "error": ""
    }
  ]
}
```

Available tool names: `web_search`, `retrieve`, `lint_code`, `format_code`, `typecheck_code`, `execute_code`

---

## Models

### `GET /api/models`

List all GGUF models discovered in the models directory.

```json
[
  {
    "name": "Qwen3.6 35B Q4_K_M",
    "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "path": "/models/qwen/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "family": "qwen",
    "parameters": "35B",
    "quantisation": "Q4_K_M",
    "context": 8192,
    "chat_template": "chatml",
    "size_gb": 22.4,
    "exists": true,
    "description": "..."
  }
]
```

### `POST /api/models/scan`

Re-scan the models directory. Returns updated list.

### `GET /api/models/active`

Return the currently configured model and provider.

### `POST /api/models/download`

Trigger an async GGUF download from HuggingFace.

**Request**
```json
{
  "repo_id":    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
  "filename":   "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  "dest_subdir": "qwen",
  "sha256":     ""
}
```

**Response** `202` — download queued.

### `GET /api/models/download/status`

Check progress of all download operations (current session only).

```json
[
  {
    "key": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "status": "downloading",
    "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "bytes_downloaded": 8589934592,
    "total_bytes": 24058482688,
    "percent": 35.7,
    "message": "8.59 / 24.06 GB"
  }
]
```

After download completes, restart the llama.cpp container to load the new model:
```bash
docker compose restart paeka-llamacpp
```

---

## Export

### `GET /api/conversations/{id}/export?format=json`
### `GET /api/conversations/{id}/export?format=markdown`

Export a single conversation as a downloadable file.

### `GET /api/export/all?format=json`
### `GET /api/export/all?format=markdown`

Export all conversations in one file.
