import { describe, it, expect, vi } from "vitest";
import { createAgentSession } from "./agent";
import { mockSseFetch } from "./test-helpers";

describe("createAgentSession", () => {
  it("rejects an empty base URL", () => {
    expect(() => createAgentSession({ baseURL: "  ", model: "x" })).toThrow(/base URL/i);
  });

  it("rejects an empty model", () => {
    expect(() =>
      createAgentSession({ baseURL: "http://localhost:8000/v1", model: "  " }),
    ).toThrow(/model/i);
  });

  it("starts with no messages", () => {
    const chat = createAgentSession({
      baseURL: "http://localhost:8000/v1",
      model: "paeka-qwen",
      fetch: mockSseFetch(["hi"]),
    });
    expect(chat.messages).toEqual([]);
  });

  it("streams a real assistant reply through the full SDK pipeline", async () => {
    const chat = createAgentSession({
      baseURL: "http://localhost:8000/v1",
      model: "paeka-qwen",
      fetch: mockSseFetch(["Hello", " there"]),
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

  it("never sends an Authorization header when no apiKey is configured", async () => {
    const fetchMock = mockSseFetch(["ok"]);
    const chat = createAgentSession({
      baseURL: "http://localhost:8000/v1",
      model: "paeka-qwen",
      fetch: fetchMock,
    });

    await chat.sendMessage({ text: "hi" });

    const [, init] = (fetchMock as ReturnType<typeof vi.fn>).mock.calls[0] as [
      string,
      RequestInit,
    ];
    const headers = new Headers(init.headers);
    expect(headers.has("Authorization")).toBe(false);
  });
});
