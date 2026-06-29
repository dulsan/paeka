import { describe, it, expect } from "vitest";
import { createAgentSession } from "./agent";
import { mockJsonFetch } from "./test-helpers";

describe("createAgentSession", () => {
  it("rejects an empty base URL", () => {
    expect(() => createAgentSession({ baseURL: "  " })).toThrow(/base URL/i);
  });

  it("starts with no messages", () => {
    const chat = createAgentSession({
      baseURL: "http://localhost:8000/v1",
      fetch: mockJsonFetch("hi"),
    });
    expect(chat.messages).toEqual([]);
  });

  it("sends a message and receives a real assistant reply through the full pipeline", async () => {
    const chat = createAgentSession({
      baseURL: "http://localhost:8000/v1",
      fetch: mockJsonFetch("Hello there"),
    });

    await chat.sendMessage({ text: "hi" });

    expect(chat.status).toBe("ready");
    expect(chat.error).toBeUndefined();
    expect(chat.messages).toHaveLength(2);
    expect(chat.messages[0].role).toBe("user");
    expect(chat.messages[1].role).toBe("assistant");

    const assistantText = chat.messages[1].parts
      .filter((p): p is { type: "text"; text: string } => p.type === "text")
      .map((p) => p.text)
      .join("");
    expect(assistantText).toBe("Hello there");
  });

  it("derives /api/agent/react from a baseURL ending in /v1", async () => {
    const fetchMock = mockJsonFetch("ok");
    const chat = createAgentSession({
      baseURL: "http://localhost:8000/v1",
      fetch: fetchMock,
    });

    await chat.sendMessage({ text: "hi" });

    const [url] = (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/agent/react");
  });

  it("never sends an Authorization header when no apiKey is configured", async () => {
    const fetchMock = mockJsonFetch("ok");
    const chat = createAgentSession({
      baseURL: "http://localhost:8000/v1",
      fetch: fetchMock,
    });

    await chat.sendMessage({ text: "hi" });

    const [, init] = (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0] as [
      string,
      RequestInit,
    ];
    const headers = new Headers(init.headers);
    expect(headers.has("Authorization")).toBe(false);
  });
});
