import type { FetchFunction } from "@ai-sdk/provider-utils";
import { Chat, type UIMessage } from "@ai-sdk/svelte";
import { ReactAgentTransport } from "./reactTransport";

/**
 * Configuration for a chat session against Paeka's own backend.
 */
export interface AgentSessionConfig {
  /**
   * Base URL of the Paeka backend, e.g. "http://localhost:8000/v1" (the
   * /v1 suffix, if present, is stripped -- kept accepting it in this shape
   * for continuity with existing deployments/saved settings that predate
   * the switch to /api/agent/react below).
   */
  baseURL: string;
  /** Optional bearer token. Omit for a local, unauthenticated instance. */
  apiKey?: string;
  /** Injectable for tests; real fetch is used if omitted. */
  fetch?: FetchFunction;
}

/**
 * [SEAM] This function is the one place that decides how chat messages
 * actually get to a model. Talks to Paeka's bespoke /api/agent/react
 * endpoint -- the LangGraph ReAct loop that actually executes MCP tools
 * server-side -- via ReactAgentTransport, rather than the generic
 * OpenAI-compatible passthrough at /v1/chat/completions (that endpoint
 * exists for external clients like Terax; this UI now talks to the
 * tool-executing agent directly so tool calls are visible here too).
 *
 * When the Tauri/Rust shell is added later (per the SAD's "Rust AI
 * orchestrator" data-flow), this is the only place that needs to
 * change -- swap ReactAgentTransport for a transport that calls
 * invoke('send_chat_message', ...) and adapts Tauri's event stream into
 * a ReadableStream<UIMessageChunk>. Every component that consumes the
 * returned Chat instance stays exactly as it is.
 */
export function createAgentSession(config: AgentSessionConfig): Chat<UIMessage> {
  const baseURL = config.baseURL.trim();
  if (!baseURL) {
    throw new Error("Connection settings: base URL is required.");
  }

  return new Chat({
    transport: new ReactAgentTransport({
      url: toReactEndpoint(baseURL),
      apiKey: config.apiKey,
      fetch: config.fetch,
    }),
  });
}

/** "http://host:8000/v1" -> "http://host:8000/api/agent/react" (also tolerates a base URL with no /v1 suffix already present). */
function toReactEndpoint(baseURL: string): string {
  const trimmed = baseURL.replace(/\/+$/, "");
  const root = trimmed.endsWith("/v1") ? trimmed.slice(0, -"/v1".length) : trimmed;
  return `${root}/api/agent/react`;
}
