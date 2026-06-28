# Research Workbench Platform -- chat interface

The first piece of RWP: a standalone Svelte chat UI, talking to any
OpenAI-compatible chat completions endpoint (Paeka's
`/v1/chat/completions` satisfies this directly). Deliberately **not**
a Tauri/Rust app yet -- see "What's deferred" below.

## Stack

- **Svelte 5** (runes) + **Vite 8** -- plain Vite, not SvelteKit. The
  eventual Tauri shell wraps a static frontend bundle with no Node
  server at runtime, so this avoids building against a server-route
  pattern that wouldn't carry over.
- **TypeScript 6.0** -- TS 7.0 only reached RC status recently and its
  tooling-facing API isn't stable until 7.1 (per Microsoft's own
  announcement, "several months" off as of when this was built). TS
  6.0 was deliberately built as a compatibility bridge with
  near-identical type-checking semantics, so the eventual bump should
  be close to a non-event.
- **Vercel AI SDK** (`ai` + `@ai-sdk/svelte` + `@ai-sdk/openai-compatible`)
  for the chat session itself, rather than hand-rolling SSE parsing
  and message-state management. `DirectChatTransport` is the piece
  that makes this work without a backend route: it talks to an `Agent`
  in-process (i.e. straight from the browser), which is exactly what a
  single-process / no-server-runtime app needs.
- **Vitest** + **@testing-library/svelte** for tests.

## Setup

This repo targets **Bun** as the package manager (per the project
notes). Built and verified here with npm, since that's what was
available in the build/verification environment -- everything in
`package.json` is plain, Bun-compatible config, so:

```bash
bun install
bun run dev      # http://localhost:5173
bun run test     # vitest run
bun run check    # svelte-check (TypeScript + template validation)
bun run build    # production build -> dist/
```

(`npm install` / `npm run <script>` work identically if you don't have
Bun yet.)

## Where the AI provider seam actually is

`src/lib/ai/agent.ts`'s `createAgentSession()` is the *only* place that
decides how chat messages reach a model. Today: a `DirectChatTransport`
wrapping an `Agent` that calls an OpenAI-compatible endpoint straight
from the browser.

The SAD's data-flow diagram routes AI calls through a Rust "AI
orchestrator," not direct frontend-to-provider calls. When the
Tauri/Rust shell gets built, that's the one function that changes --
swap `DirectChatTransport` for a custom `ChatTransport` that calls
`invoke('send_chat_message', ...)` and adapts Tauri's event stream into
a `ReadableStream<UIMessageChunk>`. Every component that consumes the
`Chat` instance this function returns (`ChatWindow.svelte` and
everything under it) stays exactly as it is -- they only know about
`chat.messages` / `chat.status` / `chat.sendMessage()`, never about
what's actually moving the bytes.

## What's deferred

No Tauri, no Rust, no SQLite, no workspace/paper-library/lab-book --
just the chat shell, on purpose. Two reasons: this environment has no
`cargo`/`rustc` at all, so any Rust written here would be unverifiable
guesswork; and Tauri is explicitly designed to wrap an
already-working Vite frontend with minimal disruption, so building the
chat UI standalone first and wrapping it later doesn't throw anything
away.

## Known harmless build notice

`vite build` prints a Node built-in externalization notice for
`diagnostics_channel` (imported somewhere inside the `ai` package).
This is Vite correctly stubbing out a Node-only module for the browser
build -- expected for an isomorphic package, not a bug, and the actual
build/test/runtime behaviour (including a full mocked streaming
round-trip in `ChatWindow.test.ts`) is verified working regardless.
