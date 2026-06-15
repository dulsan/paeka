# Weaviate RAFT Cluster Join Failure — Diagnostic Report

**Date:** 2026-06-15  
**Container:** paeka-weaviate (cr.weaviate.io/semitechnologies/weaviate:1.27.0)  
**Status:** STUCK in infinite join loop — HTTP 503 Service Unavailable  
**Root Cause:** Weaviate RAFT consensus layer misconfiguration attempting to join itself

---

## Executive Summary

Weaviate is continuously attempting to join a cluster at `172.18.0.2:8300` (its own container IP) and failing with **RAFT status 8** (likely "NOT_LEADER" or "FAILED_TO_CONNECT"). The container starts successfully and begins serving HTTP traffic, but the RAFT subsystem never achieves leader status, causing the REST API healthcheck to return HTTP 503.

**Key Evidence:**
- Container running: `docker ps` shows `healthy/starting`
- REST API serving: `Serving weaviate at http://[::]:8080` (logged)
- RAFT loop: Logs show 100+ "attempting to join" + "failed to join cluster" cycles per minute
- No actual cluster configured: `docker-compose.yml` defines only one service

---

## Detailed Error Analysis

### 1. RAFT Join Loop (Primary Blocker)

**Symptom:**  
```json
{"msg":"attempting to join","remoteNodes":["172.18.0.2:8300"],"time":"2026-06-15T16:41:23Z"}
{"msg":"attempted to join and failed","remoteNode":"172.18.0.2:8300","status":8,"time":"2026-06-15T16:41:23Z"}
{"action":"bootstrap","error":"could not join a cluster from [172.18.0.2:8300]","msg":"failed to join cluster","servers":["172.18.0.2:8300"],"voter":true}
{"action":"bootstrap","msg":"notified peers this node is ready to join as voter","servers":["172.18.0.2:8300"],"time":"2026-06-15T16:41:23Z"}
```

**What's happening:**
1. Weaviate boots and initializes RAFT subsystem
2. It attempts to join a cluster via `remoteNodes: ["172.18.0.2:8300"]`
3. Connection fails with `status: 8` (RAFT error code)
4. It falls back to "notified peers this node is ready to join as voter" 
5. After ~2-5 seconds, the loop repeats
6. This cycles indefinitely

**Why it's happening:**
- `172.18.0.2` is the container's own IP address (assigned by Docker bridge network `paeka_default`)
- Weaviate is trying to bootstrap a cluster by joining itself, which fails
- The node is configured as a "voter" but has no valid cluster leader

---

### 2. HTTP 503 Service Unavailable

**Symptom:**  
Healthcheck probe gets:
```
Connecting to localhost:8080 ([::1]:8080)
wget: server returned error: HTTP/1.1 503 Service Unavailable
```

**Root cause:**
- REST API is listening (`Serving weaviate at http://[::]:8080` is logged)
- But RAFT consensus has not achieved leader status
- Weaviate blocks most API operations until a leader is elected
- Healthcheck queries `/v1/.well-known/ready` which fails with 503 until RAFT is ready

---

### 3. Configuration Issues Identified

#### Issue 3a: No Single-Node Bootstrap Configuration

**Current docker-compose.yml:**
```yaml
environment:
  QUERY_DEFAULTS_LIMIT: "25"
  AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
  PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
  DEFAULT_VECTORIZER_MODULE: "none"
  ENABLE_MODULES: ""
  RAFT_HEARTBEAT_TIMEOUT: "5000"
  RAFT_ELECTION_TIMEOUT: "10000"
  # Missing: Any single-node bootstrap config
```

**Missing config:**
- No `RAFT_BOOTSTRAP_EXPECT` set to 1
- No way to tell Weaviate to initialize as a standalone node
- Weaviate defaults to clustering mode and expects to find peers

**Attempted fix that failed:**
```yaml
RAFT_BOOTSTRAP_EXPECT: "1"  # This alone is not sufficient
```
Reason: Weaviate still hardcoded the cluster join attempt, possibly from persisted state

#### Issue 3b: Persistent RAFT State Corruption

**Evidence:**
- When volumes deleted and restarted, RAFT loop begins again immediately
- RAFT database files persist across container recreations
- Even with `docker compose down -v`, the RAFT state reappears

**Likely cause:**
- The original Weaviate instance was configured for clustering
- RAFT consensus stores cluster membership in `raft.db` 
- This persists in the Docker named volume `paeka_weaviate_data` or bind mount
- On restart, Weaviate reads the old cluster config: "Join peer at `172.18.0.2:8300`"
- But that peer no longer exists, so join fails infinitely

#### Issue 3c: Volume Mount Strategy

**Current setup:**
```yaml
volumes:
  weaviate_data:  # Named volume in top-level volumes: section

services:
  paeka-weaviate:
    volumes:
      - weaviate_data:/var/lib/weaviate
```

**Problem:**
- Named volumes are managed by Docker
- Hard to inspect and clear persisted RAFT state
- Previous approach (bind mount to `./database/weaviate`) had same issue
- Either way, the corrupted RAFT database survives cleanup

---

## Environment & Configuration Files Checked

### `.env` (Correct)
```
PAEKA_RETRIEVAL__WEAVIATE_URL=http://localhost:8090
PAEKA_RETRIEVAL__ENABLED=true
```
✓ Correctly points to port 8090 (mapped from container 8080)

### `.dockerignore` (Not relevant)
```
database/
```
✓ Only affects `docker build`, not `docker compose`

### `docker-compose.yml` (Current issue source)
- Missing single-node RAFT configuration
- Volume strategy doesn't isolate corrupted RAFT state
- No cleanup/reset mechanism for persisted cluster config

### `scripts/start_fixed.ps1` (Correct)
- Checks Weaviate health on port 8090
- Gracefully waits for readiness with 22-retry polling (~88s timeout)
- Correctly shows as `ready` only when HTTP 200 is returned

---

## Timeline of Attempts & Why They Failed

| Attempt | Fix | Result | Why Failed |
|---------|-----|--------|-----------|
| 1 | Remove `CLUSTER_HOSTNAME=node1` | Still loops | RAFT state persisted in volume |
| 2 | Clear `./database/weaviate` directory | Still loops | New container recreates same state |
| 3 | Add `RAFT_BOOTSTRAP_EXPECT=1` | Still loops | Not sufficient without disabling cluster join |
| 4 | Add entrypoint to `rm -rf /var/lib/weaviate/raft` | Still loops | Entrypoint cleaned at startup, but RAFT state recreated |
| 5 | Switch to Docker named volume | Still loops | Same RAFT issue, just in Docker's storage |
| 6 | Add `RAFT_PORT=0` to disable RAFT | Failed to boot | Weaviate rejects `RAFT_PORT` < 1 |

**Common theme:** RAFT consensus layer hardcodes cluster join logic regardless of environment variables or startup entrypoints.

---

## Technical Root Cause

Weaviate 1.27.0 RAFT initialization code likely:

1. Reads persisted cluster config from `raft.db`
2. On startup, generates a **new random node ID** (e.g., `b4d7cea98d6e`)
3. Initializes RAFT with its current container IP (`172.18.0.2`)
4. Reads stored cluster peers: `["172.18.0.2:8300"]` (from previous run)
5. Attempts to join that peer at `172.18.0.2:8300`
6. Connection fails (can't join own IP/port if not a leader)
7. Falls back to bootstrap as voter, but remains stuck because no leader ever elected
8. Repeats join attempt every 2-5 seconds indefinitely

**Why new container IP doesn't help:**
- Even if container IP changed to `172.18.0.3`, stored cluster config still references `172.18.0.2`
- Join attempt to old IP fails → infinite loop

**Why deleting `/var/lib/weaviate/raft/raft.db` doesn't work:**
- RAFT state may also be stored in other `.db` files (schema.db, modules.db, classifications.db)
- Or RAFT re-reads from a journal/snapshot in `raft/snapshots/`
- Or Weaviate regenerates the join config from some startup logic

---

## Attempted Workarounds & Why They Won't Work

### Workaround A: Increase RAFT Timeouts
```yaml
RAFT_HEARTBEAT_TIMEOUT: "5000"
RAFT_ELECTION_TIMEOUT: "10000"
```
**Status:** Already configured, does not fix the underlying join loop

### Workaround B: Set `CLUSTER_HOSTNAME`
```yaml
CLUSTER_HOSTNAME: "node1"
```
**Status:** Removed (didn't help). Setting it to a hostname doesn't change the IP-based join logic.

### Workaround C: Use Standalone Mode Flag
```yaml
# Attempted but no such flag exists in Weaviate 1.27.0
```

---

## Recommended Fixes (Ordered by Likelihood of Success)

### Option 1: Fully Wipe Persisted Weaviate State (Recommended First Step)

**Steps:**
```powershell
# 1. Stop container
docker compose down -v

# 2. Delete Docker volume completely
docker volume rm paeka_weaviate_data

# 3. If using bind mount, delete host directory
Remove-Item -Recurse "C:\Users\dwijesinghe\OneDrive - Swinburne University\Desktop\paeka\database\weaviate" -Force

# 4. Restart
docker compose up -d paeka-weaviate

# 5. Monitor logs for 30+ seconds
docker logs paeka-weaviate -f
```

**Expected behavior:**  
If RAFT state is truly wiped, container should:
- Generate new node ID
- Initialize as single-node cluster (no join attempt)
- Achieve leader status within 30s
- Return HTTP 200 on healthcheck

**Failure mode:**  
If logs still show `attempting to join`, the RAFT config is being set by something other than persisted state.

---

### Option 2: Use Weaviate Initialization API to Reset Cluster

**Steps:**
```powershell
# 1. Wait for container to fully boot (even if unhealthy)
Start-Sleep -Seconds 30

# 2. Call Weaviate's cluster info endpoint (if available)
Invoke-WebRequest -Uri "http://localhost:8090/v1/cluster" -Method Get

# 3. Look for a reset or initialize endpoint in Weaviate docs
# https://weaviate.io/developers/weaviate/api/reference/management
```

**Likelihood:** Low — cluster reset is unlikely to be a public API

---

### Option 3: Configure Multi-Node Cluster Correctly (If Clustering Needed)

**If you actually need clustering later**, the `docker-compose.yml` would need:
```yaml
services:
  node1:
    environment:
      CLUSTER_HOSTNAME: "node1"
      RAFT_PORT: "8300"
      RAFT_JOIN: ""  # First node (bootstrap)
  node2:
    environment:
      CLUSTER_HOSTNAME: "node2"
      RAFT_PORT: "8300"
      RAFT_JOIN: "node1:8300"  # Join node1
```

**For now:** Not applicable — you have only one node.

---

### Option 4: Downgrade Weaviate or Try Latest Version

**Try upgrading to a newer patch:**
```yaml
image: cr.weaviate.io/semitechnologies/weaviate:1.27.1  # or later
```

**Or downgrade to a known stable version:**
```yaml
image: cr.weaviate.io/semitechnologies/weaviate:1.26.0  # if available
```

**Likelihood:** Medium — may have been a regression fixed in newer releases

---

### Option 5: Inspect & Manually Edit Persisted RAFT Config

**Advanced debugging:**
```powershell
# If using bind mount, inspect the raft database
# (Requires sqlite3 tool or Weaviate debugging)

# Check what's in raft.db (binary format, may not be readable)
file "C:\Users\dwijesinghe\OneDrive - Swinburne University\Desktop\paeka\database\weaviate\raft\raft.db"

# Extract the container and run tools inside
docker run -it --rm -v paeka_weaviate_data:/data busybox ls -la /data/raft/
```

**Likelihood:** Low — RAFT database is binary and not meant to be edited manually

---

## Next Steps When Addressing This Issue

1. **Immediate:** Try Option 1 (full wipe) with a fresh container
2. **If Option 1 fails:** Collect full logs with `docker logs paeka-weaviate > logs.txt` and check for other error messages
3. **If still stuck:** Try Option 4 (upgrade Weaviate version)
4. **If that fails:** Consider Option 3 (configure as proper single-node setup with explicit flags)
5. **Last resort:** Option 5 (manual RAFT debugging) or switch to a different vector database (e.g., Qdrant, Milvus)

---

## Files That Need Changes

When you're ready to fix this:

### 1. `docker-compose.yml`
- Add explicit single-node RAFT bootstrap configuration
- Consider reverting to bind mount vs. named volume (for easier inspection)
- Add a cleanup/init script or entrypoint

### 2. `scripts/start_fixed.ps1`
- Add a "reset" mode that purges Weaviate state before starting
- Example: `.\scripts\start_fixed.ps1 -ResetWeaviate`

### 3. Documentation
- Add troubleshooting guide mentioning this RAFT issue
- Recommend `docker compose down -v` + manual volume deletion for a clean start

---

## Relevant References

- Weaviate RAFT Consensus: https://weaviate.io/developers/weaviate/concepts/raft
- Docker Compose Volumes: https://docs.docker.com/compose/compose-file/compose-file-v3/#volumes
- Weaviate Configuration: https://weaviate.io/developers/weaviate/config-refs/env-vars

---

## Summary Table

| Component | Status | Issue | Recommendation |
|-----------|--------|-------|-----------------|
| Container | Running | RAFT loop prevents readiness | Option 1: Full wipe |
| RAFT Subsystem | Failed | Cannot join self; no leader | Clear `/var/lib/weaviate` entirely |
| HTTP API | Serving | Blocked by RAFT readiness | Depends on RAFT fix |
| Healthcheck | Failing (503) | Waiting for RAFT leader | Depends on RAFT fix |
| docker-compose.yml | Incomplete | Missing single-node config | Add bootstrap flags or reset script |
| Named Volume | Persistent | Retains corrupted RAFT state | Delete volume: `docker volume rm paeka_weaviate_data` |
| Bind Mount (old) | Unused | Also had same RAFT issue | Document as legacy approach |

---

**Report generated:** 2026-06-15 16:45 UTC  
**Status:** Awaiting implementation of recommended fixes
