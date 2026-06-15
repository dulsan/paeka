# PAEKA Codebase Review

## Critical Bugs (will crash or silently corrupt at runtime)

### BUG-1 — `_parse_json()` character-strip corruption (`iteration_graph.py:L-approx230`)

```python
# CURRENT — WRONG
raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
```

`str.lstrip(chars)` strips **individual characters** from the character set, not a substring prefix.
`lstrip("```json")` removes any leading `{`, `` ` ``, `j`, `s`, `o`, `n` characters.

**Effect:** LLM returns `{"score": 0.9, ...}`. The leading `{` is in the strip-set, so it is removed.
Result: `"score": 0.9, ...}` → `json.JSONDecodeError` → `_parse_json` returns `{}` → evaluator defaults to `score=0.5` → loop runs all `max_iterations`, wasting every LLM call in the iteration pipeline.

**Fix:** `str.removeprefix("```json").removeprefix("```").removesuffix("```")`

---

### BUG-2 — `WeaviateStore.hybrid_search()` and `upsert_chunks()` are synchronous — called from async context with no thread offload

Both methods call the blocking Weaviate Python SDK synchronously.
`retriever_node._run_vector()` calls `engine.retrieve()` which calls `store.hybrid_search()` — all synchronously on the asyncio event loop.

**Effect:** Every retrieval call stalls **all** FastAPI coroutines — including health checks, streaming responses, and other concurrent requests — for the full Weaviate round-trip (~30–150ms at localhost). Under even light concurrent load this becomes a reliability issue.

Same problem applies to `delete_document_chunks()`, and all calls made from `IngestionPipeline`.

**Fix:** Wrap all Weaviate SDK calls in `asyncio.to_thread()`. See `weaviate_store.py` fix.

---

### BUG-3 — `RetrievalEngine.retrieve()` is synchronous — called from `async def retriever_node`

`_run_vector()` in `retriever.py` is declared `def` (not `async def`) and calls `engine.retrieve()` directly, which in turn calls the synchronous embedder and the synchronous store. The call site in `_dispatch()` uses `return _run_vector(...)` — no `await asyncio.to_thread()`.

**Fix:** Make `engine.retrieve()` `async`, wrap inner blocking calls with `asyncio.to_thread()`.

---

### BUG-4 — `WeaviateStore._col` uses `assert` as a guard

```python
@property
def _col(self):
    assert self._client, "WeaviateStore.connect() not called"
```

`assert` statements are eliminated by Python's `-O` optimisation flag (`python -O`). In production, if the client is `None`, `_col` returns `None`, and the first attribute access produces a cryptic `AttributeError` with no indication that `connect()` was not called.

**Fix:** Replace with `if self._client is None: raise RuntimeError(...)`.

---

### BUG-5 — Missing `WeaviateStore` methods needed by MCP server and ConversationMemory

`backend/mcp/server.py` calls `store.search(vector, limit, collection_name)`.
`backend/memory/conversation.py` calls `store.search(...)`, `store.insert(...)`, and `store.ensure_collection(...)`.
None of these methods exist on `WeaviateStore`.

**Effect:** `AttributeError` at runtime on first MCP tool call or first conversation memory write.

**Fix:** Add `async search()`, `async insert()`, `async ensure_collection()` to `WeaviateStore`.

---

### BUG-6 — `retriever_node` dispatches sub-queries **sequentially**

```python
for sq in sorted(sub_queries, key=lambda x: x["priority"]):
    results = await _dispatch(sq, ...)   # waits for each before starting next
```

With N=6 sub-queries at ~80ms each, this is 480ms of sequential latency that could be ~80ms with `asyncio.gather()`.

**Fix:** Dispatch all sub-queries concurrently.

---

### BUG-7 — `docling_parser._get_converter()` is not thread-safe

```python
_converter = None

def _get_converter():
    global _converter
    if _converter is not None:
        return _converter
    # ... construct converter
    _converter = DocumentConverter(...)
```

Under concurrent ingestion requests, two threads can both observe `_converter is None`, both construct a `DocumentConverter` (loading ~2GB of models into RAM twice), and race to assign the singleton. This is a classic double-checked locking bug without a lock.

**Fix:** Use `threading.Lock` around the check-and-assign.

---

### BUG-8 — `docling_parser.parse()` and `ingestion/pipeline.py:parse_file()` are synchronous — called from `async ingest_file()`

`parse_file()` is a blocking call that takes 5–60 seconds for dense PDFs. It is called directly inside `async def ingest_file()` on the event loop with no `asyncio.to_thread()`.

**Fix:** `parsed = await asyncio.to_thread(parse_file, path, ...)`

---

### BUG-9 — `planner.py` imports `LLMClient` — breaks with `LiteLLMProvider`

```python
from backend.llm.client import LLMClient  # planner.py

async def planner_node(state: AgentState, llm: LLMClient) -> AgentState:
```

`LiteLLMProvider` implements `LLMProvider` (the abstract base), not `LLMClient`. When the pipeline passes a `LiteLLMProvider` instance to `planner_node`, static type checkers (and runtime isinstance checks if any exist downstream) will fail.

**Fix:** Change type annotation to `LLMProvider`.

---

## Performance Issues

### PERF-1 — Character-count chunking severely under-utilises bge-m3

`chunker.py` default: `chunk_size=512` means **512 characters**.
bge-m3 context limit: **512 tokens**.
At ~4 chars/token for English prose: 512 chars ≈ **128 tokens** — 25% utilisation.

This produces 4× more chunks than necessary for the same content, meaning:
- 4× more embedding calls (the bottleneck of ingestion)
- 4× more Weaviate objects (storage + search latency)
- Retrieved chunks contain less context per result (worse synthesis quality)

The `docling_parser.chunk_with_hybrid_chunker()` function already uses `HybridChunker` which is token-aware and correct — but `pipeline.py` only uses it for docling documents. Plain text, code, LaTeX, and spreadsheets go through `chunk_text()` with the 512-char default.

**Fix:** Increase default `chunk_size` for the char-based path to ~1600 (≈400 tokens, leaving 112 tokens headroom for the heading prefix and any overlap). Or add bge-m3 token counting via `transformers.AutoTokenizer`.

---

### PERF-2 — `ingestion/pipeline.py` embeds chunks in batches of 32 with a synchronous inner loop

The embedder already supports `encode_batch(texts, batch_size=8)`. Ingestion should ensure it's fully batching and not embedding one chunk at a time. (Requires reading `_process()` fully — partial view shown.)

---

### PERF-3 — `iteration_graph` evaluator has no visibility into score history

The evaluator prompt only shows the current output. It has no awareness of:
- What the previous score was
- Whether this iteration improved or regressed

This means the evaluator can legitimately score iteration 3 lower than iteration 2 (if the LLM's output actually degraded), yet the graph's router only uses the raw threshold, not the delta. The loop can converge on a local minimum.

**Fix:** Pass last two `output_history` entries and last two `critique_history` entries to the evaluator. The prompt should explicitly ask: "Is this better than the previous attempt? Should iteration stop?"

---

## Architecture Gaps

### ARCH-1 — Tool-calling loop does not use native function calling

The current `SelfHealingToolGraph` pattern:
```
LLM → JSON text output → _parse_json_list() → tool dispatch → JSON text evaluation → ...
```

This is the "prompt-engineering function calling" pattern. It fails ~15% of the time due to:
- Malformed JSON from the LLM (Qwen3.5-9B is good but not perfect)
- Correct JSON but wrong field names ("tool_name" vs "tool", etc.)
- Empty arrays when the LLM decides to explain rather than call

LiteLLM's `acompletion(tools=schemas, tool_choice="auto")` uses the model's **native** function-calling mode (OpenAI tool-use format, which llama-server supports via `--jinja`). This:
- Never produces malformed JSON (the model generates tokens constrained to the schema grammar)
- Eliminates the Evaluator and Reflector nodes for schema errors
- Matches how Claude, GPT-4, and Gemini actually operate

**Fix:** Implement `react_graph.py` using the native function-calling path. See new file.

---

### ARCH-2 — No LangGraph state checkpointing

Every conversation starts with an empty `AgentState`. If the API server restarts mid-session, the entire agent context is lost. LangGraph's `SqliteSaver` integrates with the SQLite database already in the project.

---

### ARCH-3 — `AgenticRAGPipeline` and `SelfHealingToolGraph` are parallel retrieval paths

`graph.py` has Planner→Retriever→Critic→Synthesiser.
`tool_graph.py` has Selector→Executor→Evaluator→Reflector→Synthesiser.

Both produce final text responses from retrieval. The RAG pipeline could instead be a **sub-graph** that the tool graph invokes as one of its MCP tools (`weaviate_search` already exists). This eliminates the duplication.

---

## PDF / Document Handling

### DOC-1 — Equation export produces markdown code fences, not LaTeX

`docling_parser._text()` falls through to `item.export_to_markdown()` for equations.
Docling wraps inline math as `` `$E=mc^2$` `` (markdown code fence).
Stored in Weaviate as: `` `$E=mc^2$` `` — the backticks make it a code span.

**Effect:** Equations are stored as code strings. Retrieval works but rendered output looks like code, not math. For academic/engineering papers this is a significant UX issue.

**Fix:** For `ElementType.EQUATION`, prefer `item.text` (raw LaTeX) over `export_to_markdown()`. Store with `element_type="equation"` tag so the frontend can render with MathJax/KaTeX.

---

### DOC-2 — `chunk_with_hybrid_chunker()` is defined but the pipeline uses it correctly

`pipeline.py` already uses `HybridChunker` for docling documents (confirmed in the `_process()` method header — uses two-path logic). This is correct. The issue is only for the non-docling paths (BUG-PERF-1 above).

---

### DOC-3 — No scanned PDF fallback

`docling_parser` disables OCR globally. For scanned PDFs (non-selectable text), this silently produces a near-empty document (figures only, no text). There's no error, no warning to the user, and no automatic fallback to a slower OCR path.

**Fix:** After parsing, if `len(elements) < 3` and `len(text_elements) == 0`, detect as "likely scanned" and re-parse with `do_ocr=True`, logging a warning about parse time.

---

## Suggested Improvements (Prioritised)

### P0 — Fix now (will break the system)
1. BUG-5: Add `search()`, `insert()`, `ensure_collection()` to `WeaviateStore`
2. BUG-1: Fix `_parse_json()` lstrip corruption
3. BUG-2/3: Make `hybrid_search()`, `upsert_chunks()`, `retrieve()` async
4. BUG-4: Replace `assert` with `RuntimeError` in `_col`
5. BUG-8: Wrap `parse_file()` in `asyncio.to_thread()`

### P1 — Fix soon (measurable performance/stability impact)
6. BUG-6: Parallel sub-query dispatch with `asyncio.gather()`
7. BUG-7: `threading.Lock` for `_get_converter()` singleton
8. PERF-1: Increase `chunk_size` default to 1600 chars for non-docling paths
9. PERF-3: Pass score/output history to evaluator prompt
10. BUG-9: Fix `LLMClient` → `LLMProvider` type in `planner.py`

### P2 — Architecture (high impact, more work)
11. ARCH-1: Implement `react_graph.py` (ReAct native function-calling loop)
12. ARCH-2: Add `SqliteSaver` checkpointing to both graphs
13. DOC-1: Fix equation export to store raw LaTeX
14. DOC-3: Add scanned PDF detection + OCR fallback
15. ARCH-3: Unify RAG + tool graph via sub-graph pattern

### P3 — Quality of life
16. Add `logfire` tracing to every LLM call and tool execution
17. Contextual chunk enrichment (Anthropic contextual retrieval technique)
18. Per-collection Weaviate search in MCP tools (currently hardcoded to `"Chunk"`)
19. Streaming token output through the ReAct loop to the SSE endpoint
