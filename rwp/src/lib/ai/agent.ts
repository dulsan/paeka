import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { FetchFunction } from "@ai-sdk/provider-utils";
import { ToolLoopAgent, DirectChatTransport } from "ai";
import { Chat, type UIMessage } from "@ai-sdk/svelte";

/**
 * Configuration for a chat session against any OpenAI-compatible chat
 * completions endpoint (Paeka's /v1/chat/completions satisfies this
 * directly -- it proxies llama.cpp's own OpenAI-formatted SSE chunks
 * through unchanged, see backend/api/routes/openai_compat.py).
 */
export interface AgentSessionConfig {
  /** Base URL up to and including /v1, e.g. "http://localhost:8000/v1". */
  baseURL: string;
  /** Model id as the backend expects it, e.g. "paeka-qwen". */
  model: string;
  /** Optional bearer token. Omit for backends that don't require one
   *  (e.g. a local Ollama-backed Paeka instance). */
  apiKey?: string;
  /**
   * Optional system prompt override. Left undefined by default
   * deliberately -- Paeka already injects its own configured system
   * prompt server-side (see backend/llm/ollama.py's _inject_system())
   * when the first message isn't already a system message. Setting
   * this here would silently override that, which is more surprising
   * than useful for a "simple chat interface" talking to a backend
   * that already has its own persona configured.
   */
  instructions?: string;
  /** Injectable for tests; real fetch is used if omitted. */
  fetch?: FetchFunction;
}

/**
 * [SEAM] This function is the one place that decides how chat messages
 * actually get to a model. Today: a DirectChatTransport calling an
 * OpenAI-compatible endpoint straight from the browser (no server
 * route -- this is a plain Vite SPA, not SvelteKit, since the eventual
 * Tauri shell wraps a static frontend with no Node runtime at all).
 *
 * When the Tauri/Rust shell is added later (per the SAD's "Rust AI
 * orchestrator" data-flow), this is the only place that needs to
 * change -- swap DirectChatTransport for a custom ChatTransport that
 * calls invoke('send_chat_message', ...) and adapts Tauri's event
 * stream into a ReadableStream<UIMessageChunk>. Every component that
 * consumes the returned Chat instance stays exactly as it is.
 */
export function createAgentSession(config: AgentSessionConfig): Chat<UIMessage> {
  const baseURL = config.baseURL.trim();
  const model = config.model.trim();

  if (!baseURL) {
    throw new Error("Connection settings: base URL is required.");
  }
  if (!model) {
    throw new Error("Connection settings: model is required.");
  }

  const provider = createOpenAICompatible({
    name: "rwp-openai-compatible",
    baseURL,
    apiKey: config.apiKey,
    fetch: config.fetch,
  });

  const agent = new ToolLoopAgent({
    model: provider(model),
    instructions: config.instructions,
  });

  // [NOTE] TS generic-variance friction, not a runtime issue: with no
  // `tools` configured, ToolLoopAgent infers an empty tool set, which
  // produces a UIMessage sub-type TS won't unify with Chat's broader
  // default UITools generic. Runtime correctness is verified directly
  // by agent.test.ts/ChatWindow.test.ts (full streaming round-trip
  // through the real pipeline), not just asserted away here.
  return new Chat({
    transport: new DirectChatTransport({ agent }),
  }) as unknown as Chat<UIMessage>;
}
