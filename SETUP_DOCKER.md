# PAEKA — Docker Setup

One command launches the whole agent: Qdrant, Ollama (GPU), and the PAEKA
API, wired together and talking to each other automatically.

```bash
docker compose up -d --build
```

Then visit `http://localhost:8000`. First start will take a while — Ollama
needs to import your GGUF (`paeka-ollama-init`) and the API container needs
to download the bge-m3 / bge-reranker-large weights into its `hf_cache`
volume. Subsequent starts are fast.

This setup works with any Docker engine — Docker Desktop, a bare rootless
Docker Engine, or Podman. The sections below default to the rootless,
no-Desktop path because it's lower overhead, but **if you're already on
Docker Desktop, skip straight to "Using Docker Desktop" below** — don't
go install a second engine alongside it.

### Using Docker Desktop

If Docker Desktop is already installed and `docker compose up` mostly
worked for you, there's nothing else to set up:

- GPU passthrough is handled by Desktop's own WSL2 GPU support — enable
  it under **Settings → Resources → GPU**, and make sure you have the
  Windows-side NVIDIA driver installed (not one installed from inside a
  WSL distro). No NVIDIA Container Toolkit step needed.
- Desktop isn't rootless and there isn't a supported way to make it one —
  that's a property of the Engine, not something Desktop exposes. If that
  matters to you, see the no-Desktop path below instead.
- Everything else in this doc (volumes, day-to-day commands,
  troubleshooting) applies the same either way.

The rest of "Host prerequisites" below is for the no-Desktop path —
skip it if Desktop is already working for you.

---

## 1. Host prerequisites (one-time)

### Docker Engine, no Docker Desktop

**On a Linux host (including WSL2):**

```bash
curl -fsSL https://get.docker.com | sh
```

This installs the Engine + CLI + Compose plugin directly — no Desktop GUI,
no VM layer on top of your VM (if you're already in WSL2).

**On Windows:** install WSL2 if you haven't already (`wsl --install`), then
run the line above *inside* your WSL2 distro (e.g. Ubuntu), not in
PowerShell. This is the actual mechanism that lets you drop Docker Desktop
entirely on Windows — Desktop's main job was bridging Windows to a Linux
VM running the real Engine; running the Engine straight inside WSL2 skips
that bridge.

### Rootless mode

```bash
# still as your normal user, after the Engine install above
dockerd-rootless-setuptool.sh install
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
```

Add that `export` line to your shell profile (`.bashrc` / `.zshrc`) so it
persists. Verify with `docker info` — it should show `Context: rootless`.

**Honest caveat:** rootless Docker + NVIDIA GPU passthrough is the one
combination that still has real rough edges upstream (device-node
visibility inside a rootless daemon's user namespace varies by driver
version and distro). If you hit `could not select device driver` or
`permission denied` specifically on the `paeka-ollama` container after
following the GPU steps below, that's the known friction point, not
something wrong with this compose file. Two ways through it:

- Easiest: run a normal (non-rootless) Docker Engine instead, just skip
  the `dockerd-rootless-setuptool.sh` step above. You still get "no Docker
  Desktop," you lose the rootless *engine* property specifically.
- Or: try [Podman](https://podman.io) instead of Docker — it's rootless
  by default and has more mature CDI-based GPU support as of 2026. The
  compose file works unchanged with `podman compose up -d --build`.

### NVIDIA Container Toolkit (for GPU passthrough)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker   # or: systemctl --user restart docker (rootless)
```

Verify before touching compose at all:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If that doesn't show your GPU, fix it here first — `docker compose up`
will fail at the same step with a less obvious error.

---

## 2. First launch

Before the first `up`, make sure nothing native is still holding the ports
this stack needs. On Windows PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 6333,11434,8000 -ErrorAction SilentlyContinue
```

If that returns anything, find the owning process (`Get-Process -Id <PID>`)
and stop it — most commonly a native `ollama serve` left running from a
background/tray auto-start, or a leftover `qdrant.exe` / `uv run python
main.py` from an earlier native session.

```bash
git pull        # make sure you have the docker-compose.yml from this fix
docker compose up -d --build
docker compose logs -f paeka-api   # watch startup
```

What happens, in order:
1. `paeka-qdrant` starts (named volume, not a bind mount — Qdrant's own
   docs flag bind-mounted storage on Windows/WSL as a known data-loss
   risk, so this avoids that class of problem entirely).
2. `paeka-ollama` starts with GPU access and waits for `ollama list` to
   succeed.
3. `paeka-ollama-init` runs once, importing `models/qwen/Modelfile` as
   `paeka-qwen`, then exits (exit code 0 is expected and fine).
4. `paeka-api` starts only after both of the above are actually healthy.

## 3. Day-to-day

```bash
docker compose up -d          # start (no rebuild)
docker compose down            # stop, keep data
docker compose logs -f paeka-api
docker compose build paeka-api && docker compose up -d paeka-api   # rebuild just the API after a code change
```

Your conversations/SQLite/uploaded documents live in the `paeka_data` and
`paeka_database` named volumes, not in a host folder. To back them up:

```bash
docker run --rm -v paeka_database:/data -v "$(pwd)":/backup busybox \
  tar czf /backup/paeka-db-backup.tar.gz -C / data
```

## 4. Code execution sandbox (`execute_code`)

The `execute_code` tool shells out to `docker run` -- when `paeka-api` runs
inside this compose stack, it needs a way to reach a Docker daemon at all,
plus (ideally) a hardened runtime so sandboxed code doesn't share the host
kernel. Two independent pieces, both already wired into this compose file:

### 4a. Reaching Docker at all -- via a socket proxy, not a raw mount

Mounting `/var/run/docker.sock` straight into `paeka-api` would work, but
it's effectively root-equivalent access to your host for *any* process
running in that container -- not a trade-off to make implicitly. Instead,
`docker-socket-proxy` holds the real socket, and only exposes a narrow
slice of the Docker API to `paeka-api` over the internal network
(`DOCKER_HOST=tcp://docker-socket-proxy:2375`): create/start containers,
pull the python/alpine/node sandbox images, and `docker info`. Everything
else -- `exec` into arbitrary containers, touching volumes/networks
directly, building images from an arbitrary Dockerfile, swarm, secrets --
stays off. See the `docker-socket-proxy` service definition in
`docker-compose.yml` for the exact list and why each one is on/off.

Honest caveat: this filters by API *category*, not by deep-inspecting the
JSON body of a container-create call -- a real, meaningful narrowing, not
a perfect one. If `paeka-api` were ever compromised, the proxy limits what
it could do; it doesn't guarantee a create-call can't request something
within an allowed category that's more than intended.

This is on by default and needs nothing further from you -- `docker
compose up -d --build` brings up the proxy alongside everything else.

### 4b. The named-volume detail (already handled, just so you know why)

`paeka-api`'s own filesystem isn't the real Docker host's filesystem.
Writing a sandbox script to a plain tempdir inside `paeka-api` and asking
the daemon (via the proxy) to bind-mount that path into a sibling
container wouldn't work -- the daemon resolves that path against the
actual host, not against `paeka-api`'s container, and finds nothing
there. The fix is a Docker *named volume* (`paeka_sandbox_scratch`,
declared in `docker-compose.yml`'s top-level `volumes:`), mounted into
`paeka-api` and referenced *by name* in the sibling container's mount --
resolved by the daemon itself, identically regardless of host OS or path
translation. `PAEKA_SANDBOX__SCRATCH_DIR`/`PAEKA_SANDBOX__SCRATCH_VOLUME`
in `paeka-api`'s environment point at this; you shouldn't need to touch
either. See `backend/agent/sandbox.py`'s module docstring for the full
explanation.

### 4c. In-container hardening (seccomp, non-root UID, ulimits)

Rather than a separate hardened runtime (gVisor/Kata), the sandbox goes
deeper on the layer it already had -- no extra software to install, no
daemon restart, nothing host-level. This was a deliberate choice given
the actual threat model here: a single-user tool running LLM-generated
code, not an adversarial multi-tenant service. A kernel-escape 0-day is
the one thing a separate hardened runtime protects against that this
doesn't -- a real but low-probability risk for this use case, traded
off here against the operational cost of installing and maintaining a
second runtime. If that trade-off ever stops feeling right, gVisor
(`runsc`) is a drop-in `--runtime` flag away and doesn't need anything
like Kata's hypervisor/guest-kernel management -- worth asking about
again if the threat model changes (e.g. if this ever becomes
multi-tenant).

What's actually on every sandboxed container now, on top of what was
already there (`--network none`, `--read-only`, `--cap-drop ALL`,
`--security-opt no-new-privileges`, the existing memory/cpu/pids
limits):

- **A trimmed seccomp profile** (`backend/agent/sandbox-seccomp.json`)
  -- Docker's own verified default allowlist (~440 syscalls) with the
  ~53 syscalls gated behind capabilities removed. Since every
  capability is already dropped, this changes nothing about what a
  normal script can do, but means a future slip that re-adds a
  capability (e.g. someone debugging adds `--cap-add SYS_PTRACE` and
  forgets to remove it) still can't reach `ptrace`, `mount`, `unshare`,
  `setns`, kernel module loading, etc. -- seccomp blocks them
  independently of whatever capabilities happen to be held.
- **`--user 65534:65534`** -- sandboxed code now runs as the
  conventional "nobody" UID instead of root-in-container.
- **`--ulimit nproc` / `--ulimit nofile`** -- a second, independently-
  enforced process-count limit (rlimit, alongside the existing
  cgroup-based `--pids-limit`) and an open-file-descriptor cap.

Nothing to install or configure here -- this is already active. If a
sandboxed script ever fails with something like `Operation not
permitted` that didn't happen before, it's almost certainly the
seccomp profile excluding a syscall a *specific* script genuinely
needs (rather than something dangerous) -- check `docker compose logs
paeka-api` for the actual error and let me know what the script was
trying to do; loosening the profile for one specific verified-safe
syscall is a quick, low-risk fix.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ports are not available: ... 127.0.0.1:11434` | Something native already owns that port -- usually a native Ollama install auto-starting as a background/tray service. As of this fix, `paeka-ollama`'s host port isn't published at all (the API reaches it over the internal Docker network), so this specific conflict shouldn't recur. If you deliberately re-added a `ports:` mapping for host-side access, pick a free port or stop the native service first. |
| Everything in `test-paeka-api.ps1` says "connection refused" on `localhost:8000`, but `docker compose up` reported no error of its own | Check `docker compose ps` — if `paeka-api` isn't `Up`, something earlier in its `depends_on` chain (`paeka-qdrant`, `paeka-ollama-init`) failed or never went healthy, so the API container never started at all. Scroll up in the compose output for the actual root error (often a port conflict, see above) rather than trusting the test script's generic "connection refused," which just means nothing's listening. |
| `paeka-ollama` never goes healthy | GPU toolkit not configured — see step 1, verify with the bare `docker run --gpus all ... nvidia-smi` test first |
| `paeka-ollama-init` exits non-zero | Check `models/qwen/Qwen3.5-9B-Q4_K_M.gguf` actually exists on the host — it's gitignored and not committed, so a fresh clone won't have it |
| `could not select device driver` | NVIDIA Container Toolkit not installed/configured for this daemon, or you restarted the wrong daemon (rootless vs system) after `nvidia-ctk runtime configure`. **On Docker Desktop**, skip the toolkit install entirely — enable Settings → Resources → GPU support instead, and make sure it's the Windows-side NVIDIA driver installed, not a WSL-internal one. |
| API can't reach Ollama/Qdrant | Confirm you're not also running the native `qdrant.exe` / `ollama serve` on the same ports — `127.0.0.1:6333` published by compose will conflict with anything already bound there |
| `Sandbox status` still shows `docker_available: false` after `docker compose up` | Check `docker compose logs docker-socket-proxy` for startup errors first, then `docker compose logs paeka-api \| grep -i sandbox`. If the proxy container isn't `Up` at all, `paeka-api` started before it was ready -- `docker compose restart paeka-api` once the proxy is confirmed running. |
| `execute_code` returns a Docker API error mentioning permission/forbidden, but `docker_available: true` | The socket-proxy is reachable but denied a specific call -- check it isn't trying to do something outside the allowed categories (see 4a). Re-check `docker compose logs docker-socket-proxy` for which endpoint got rejected. |
| A sandboxed script fails with `Operation not permitted` that didn't fail before | The seccomp profile (4c) is excluding a syscall that specific script needs. Check `docker compose logs paeka-api` for what it was trying to do -- loosening the profile for one verified-safe syscall is a quick fix. |
