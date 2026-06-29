import type { FetchFunction } from "@ai-sdk/provider-utils";
import type { ChatRequestOptions, ChatTransport, UIMessage, UIMessageChunk } from "ai";

export interface ReactTransportConfig {
  /** Full URL to the endpoint, e.g. "http://localhost:8000/api/agent/react". */
  url: string;
  /** Optional bearer token, forwarded only if set. */
  apiKey?: string;
  /** Injectable for tests; real fetch is used if omitted. */
  fetch?: FetchFunction;
}

interface ReactHistoryTurn {
  role: "user" | "assistant";
  content: string;
}

interface ToolCallTrace {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result: string;
  ok: boolean;
}

interface ReactApiResponse {
  response: string;
  tool_calls?: ToolCallTrace[];
}

/**
 * [SEAM] Talks to Paeka's own /api/agent/react endpoint (the LangGraph
 * ReAct loop that actually executes MCP tools server-side) instead of
 * the generic OpenAI-compatible passthrough at /v1/chat/completions.
 * That endpoint is a single request -> single JSON response, not a
 * token stream, so this transport synthesizes one UIMessageChunk
 * sequence from the whole response (tool calls, then final text) rather
 * than relaying live deltas. The trade made here -- visible tool calls
 * instead of live token-by-token streaming for this turn type -- was a
 * deliberate choice, not an oversight; see PR description for the
 * alternative considered (SSE streaming from react_graph.py) and why it
 * was deferred.
 */
export class ReactAgentTransport implements ChatTransport<UIMessage> {
  constructor(private readonly config: ReactTransportConfig) {}

  async sendMessages(
    options: {
      messages: UIMessage[];
      abortSignal: AbortSignal | undefined;
    } & ChatRequestOptions,
  ): Promise<ReadableStream<UIMessageChunk>> {
    const fetchFn = this.config.fetch ?? fetch;
    const history = toHistory(options.messages);

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers instanceof Headers
        ? Object.fromEntries(options.headers.entries())
        : options.headers ?? {}),
    };
    if (this.config.apiKey) {
      headers.Authorization = `Bearer ${this.config.apiKey}`;
    }

    const res = await fetchFn(this.config.url, {
      method: "POST",
      headers,
      body: JSON.stringify({ messages: history, ...options.body }),
      signal: options.abortSignal,
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`Agent request failed (${res.status}): ${detail || res.statusText}`);
    }

    const data = (await res.json()) as ReactApiResponse;
    return toChunkStream(data);
  }

  async reconnectToStream(): Promise<ReadableStream<UIMessageChunk> | null> {
    // The endpoint is a single synchronous request/response -- there is
    // no server-side stream to resume after the fact.
    return null;
  }
}

/**
 * Reconstructs the {role, content} history the backend expects from the
 * UIMessage[] the AI SDK keeps client-side. Only text parts are sent --
 * the backend doesn't need (and shouldn't need to re-parse) a previous
 * turn's already-rendered tool-call trace to continue the conversation,
 * only what was actually said.
 */
function toHistory(messages: UIMessage[]): ReactHistoryTurn[] {
  return messages
    .filter((m): m is UIMessage & { role: "user" | "assistant" } =>
      m.role === "user" || m.role === "assistant",
    )
    .map((m) => ({
      role: m.role,
      content: m.parts
        .filter((p): p is { type: "text"; text: string } => p.type === "text")
        .map((p) => p.text)
        .join(""),
    }))
    .filter((turn) => turn.content.length > 0);
}

function toChunkStream(data: ReactApiResponse): ReadableStream<UIMessageChunk> {
  const chunks: UIMessageChunk[] = [{ type: "start" }, { type: "start-step" }];

  for (const call of data.tool_calls ?? []) {
    chunks.push({
      type: "tool-input-available",
      toolCallId: call.id,
      toolName: call.name,
      input: call.args,
      dynamic: true,
    });
    chunks.push(
      call.ok
        ? {
            type: "tool-output-available",
            toolCallId: call.id,
            output: call.result,
            dynamic: true,
          }
        : {
            type: "tool-output-error",
            toolCallId: call.id,
            errorText: call.result,
            dynamic: true,
          },
    );
  }

  // Single id is fine -- this is one non-streamed text segment per
  // response, never multiple concurrent text parts.
  const textId = "response";
  chunks.push(
    { type: "text-start", id: textId },
    { type: "text-delta", id: textId, delta: data.response },
    { type: "text-end", id: textId },
    { type: "finish-step" },
    { type: "finish" },
  );

  return new ReadableStream<UIMessageChunk>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(chunk);
      }
      controller.close();
    },
  });
}
