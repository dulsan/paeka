import { describe, it, expect, vi } from "vitest";
import { ReactAgentTransport } from "./reactTransport";
import type { UIMessage, UIMessageChunk } from "ai";

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? "OK" : "Error",
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

async function collect(stream: ReadableStream<UIMessageChunk>): Promise<UIMessageChunk[]> {
  const reader = stream.getReader();
  const out: UIMessageChunk[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.push(value);
  }
  return out;
}

function userMessage(text: string): UIMessage {
  return { id: "u1", role: "user", parts: [{ type: "text", text }] };
}

function assistantMessage(text: string): UIMessage {
  return { id: "a1", role: "assistant", parts: [{ type: "text", text }] };
}

describe("ReactAgentTransport", () => {
  it("posts the reconstructed {role, content} history to the configured url", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ response: "hi", tool_calls: [] }));
    const transport = new ReactAgentTransport({
      url: "http://localhost:8000/api/agent/react",
      fetch: fetchMock as unknown as typeof fetch,
    });

    await transport.sendMessages({
      messages: [userMessage("hello"), assistantMessage("hi there"), userMessage("again")],
      abortSignal: undefined,
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/agent/react");
    const body = JSON.parse(init.body as string);
    expect(body.messages).toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi there" },
      { role: "user", content: "again" },
    ]);
  });

  it("omits the Authorization header when no apiKey is configured", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ response: "hi", tool_calls: [] }));
    const transport = new ReactAgentTransport({
      url: "http://localhost:8000/api/agent/react",
      fetch: fetchMock as unknown as typeof fetch,
    });

    await transport.sendMessages({ messages: [userMessage("hi")], abortSignal: undefined });

    const [, init] = fetchMock.mock.calls[0];
    expect((init.headers as Record<string, string>).Authorization).toBeUndefined();
  });

  it("includes a bearer Authorization header when an apiKey is configured", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ response: "hi", tool_calls: [] }));
    const transport = new ReactAgentTransport({
      url: "http://localhost:8000/api/agent/react",
      apiKey: "secret",
      fetch: fetchMock as unknown as typeof fetch,
    });

    await transport.sendMessages({ messages: [userMessage("hi")], abortSignal: undefined });

    const [, init] = fetchMock.mock.calls[0];
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer secret");
  });

  it("synthesizes tool-input-available/tool-output-available chunks before the final text", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        response: "The answer is 4.",
        tool_calls: [
          { id: "call_1", name: "calculator", args: { expr: "2+2" }, result: "4", ok: true },
        ],
      }),
    );
    const transport = new ReactAgentTransport({
      url: "http://localhost:8000/api/agent/react",
      fetch: fetchMock as unknown as typeof fetch,
    });

    const stream = await transport.sendMessages({
      messages: [userMessage("what is 2+2?")],
      abortSignal: undefined,
    });
    const chunks = await collect(stream);
    const types = chunks.map((c) => c.type);

    expect(types).toEqual([
      "start",
      "start-step",
      "tool-input-available",
      "tool-output-available",
      "text-start",
      "text-delta",
      "text-end",
      "finish-step",
      "finish",
    ]);

    const toolInput = chunks.find((c) => c.type === "tool-input-available") as Extract<
      UIMessageChunk,
      { type: "tool-input-available" }
    >;
    expect(toolInput.toolName).toBe("calculator");
    expect(toolInput.input).toEqual({ expr: "2+2" });

    const textDelta = chunks.find((c) => c.type === "text-delta") as Extract<
      UIMessageChunk,
      { type: "text-delta" }
    >;
    expect(textDelta.delta).toBe("The answer is 4.");
  });

  it("emits tool-output-error (not tool-output-available) when a tool call failed", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        response: "I couldn't look that up.",
        tool_calls: [
          { id: "call_1", name: "web_search", args: {}, result: "[MCP ERROR] timeout", ok: false },
        ],
      }),
    );
    const transport = new ReactAgentTransport({
      url: "http://localhost:8000/api/agent/react",
      fetch: fetchMock as unknown as typeof fetch,
    });

    const stream = await transport.sendMessages({
      messages: [userMessage("look this up")],
      abortSignal: undefined,
    });
    const chunks = await collect(stream);

    expect(chunks.some((c) => c.type === "tool-output-error")).toBe(true);
    expect(chunks.some((c) => c.type === "tool-output-available")).toBe(false);
    const errorChunk = chunks.find((c) => c.type === "tool-output-error") as Extract<
      UIMessageChunk,
      { type: "tool-output-error" }
    >;
    expect(errorChunk.errorText).toContain("MCP ERROR");
  });

  it("throws with response detail when the request fails", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ detail: "model not loaded" }, false, 503));
    const transport = new ReactAgentTransport({
      url: "http://localhost:8000/api/agent/react",
      fetch: fetchMock as unknown as typeof fetch,
    });

    await expect(
      transport.sendMessages({ messages: [userMessage("hi")], abortSignal: undefined }),
    ).rejects.toThrow(/503/);
  });

  it("reconnectToStream always returns null (no resumable server-side stream)", async () => {
    const transport = new ReactAgentTransport({ url: "http://localhost:8000/api/agent/react" });
    await expect(transport.reconnectToStream()).resolves.toBeNull();
  });
});
