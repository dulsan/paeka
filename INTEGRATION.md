# PAEKA Patch — Integration Guide

## Files in this archive

```
scripts/start.ps1                          ← replace your existing file
backend/api/routes/chat_control.py         ← new file, add to your router includes
INTEGRATION.md                             ← this file
```

---

## 1. start.ps1

Drop-in replacement. Key changes made:

| Issue | Fix |
|---|---|
| `DeprecationWarning: websockets.legacy` | Uvicorn now launched with `--ws wsproto` |
| Log handlers wired but never attached | `Register-ObjectEvent` with `MessageData` now captures stdout/stderr to `logs\llama-server.log` |
| Timeout message said "30-120s" but loop ran 180s | Message corrected |
| Flash Attention not enabled | `--flash-attn` probed against binary's `--help`; added if supported, skipped if not |
| `--slots` flag missing | Added — required for `/slots/{id}` erase endpoint |
| Health probe only tried `/health` | Now tries `/health` then `/v1/models` as fallback |

### Prerequisite: install wsproto

Run once in your venv before starting:

```powershell
uv pip install wsproto
```

### Flash Attention switch

Flash Attention is **enabled by default** if your llama-server binary supports it.
To disable (e.g. for debugging):

```powershell
.\scripts\start.ps1 -NoFlashAttn
```

---

## 2. chat_control.py — wiring into main.py

Add **two imports** and **two calls** to your `main.py`:

```python
# At the top of main.py, with your other router imports:
from backend.api.routes.chat_control import (
    router as chat_control_router,
    configure as configure_chat,
)

# Inside your router includes (wherever app.include_router is called):
app.include_router(chat_control_router)

# Inside your FastAPI lifespan startup block, after LLM settings are resolved:
# Replace "http://localhost:8080" with however you read your llama port from config.
configure_chat(llama_base_url="http://localhost:8080")
```

Typical lifespan pattern:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    configure_chat(llama_base_url=settings.llm.base_url)   # ← add this
    yield
    # --- shutdown ---

app = FastAPI(lifespan=lifespan)
app.include_router(chat_control_router)                     # ← add this
```

---

## 3. New API surface

Once wired in, the following endpoints are live:

### Reset context (most common operation)

```http
POST /api/chat/reset
Content-Type: application/json

{}
```

Clears the active session's message history **and** erases the llama.cpp KV-cache
slot. The model stays loaded on the GPU — this is equivalent to starting a fresh
conversation without any restart.

Example response:
```json
{
  "ok": true,
  "session_id": "3fa85f64-...",
  "slot_erased": true,
  "slot_detail": "slot 0 erased",
  "message": "Context cleared for session 3fa85f64-..."
}
```

### Session management

```http
POST   /api/chat/sessions                         # create new session
GET    /api/chat/sessions                         # list all sessions
GET    /api/chat/sessions/{id}                    # get one session
DELETE /api/chat/sessions/{id}                    # delete + erase slot
POST   /api/chat/sessions/{id}/activate           # switch active session
```

### Using session history in your chat handler

Import the `session_store` singleton wherever you build the messages list:

```python
from backend.api.routes.chat_control import session_store

# Append messages as they arrive
session_store.append_message(session_id, role="user",      content=user_text)
session_store.append_message(session_id, role="assistant", content=reply_text)

# Retrieve history for the next completion call
history = session_store.get_history(session_id)
# → [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
```

---

## 4. Verifying the /slots endpoint

The context reset depends on llama-server being started with `--slots`.
The updated `start.ps1` adds this flag automatically.

To confirm it's working:

```powershell
# Should return {"id_slot":0,"is_processing":false,"kv_size":...}
Invoke-WebRequest -Uri "http://localhost:8080/slots" -UseBasicParsing | Select-Object -Expand Content

# Erase slot 0 manually
Invoke-WebRequest -Uri "http://localhost:8080/slots/0" -Method Post `
    -ContentType "application/json" -Body '{"action":"erase"}' `
    -UseBasicParsing | Select-Object -Expand Content
```

If `/slots` returns 404, your llama.cpp binary predates slot support
(build b2835 or earlier). In that case `slot_erased` will be `false` in
the reset response, but the in-memory session history is still cleared —
the context window itself won't be freed until the next natural expiry.
