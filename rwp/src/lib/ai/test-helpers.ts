import { vi } from "vitest";

/**
 * Builds a fetch mock that returns a real OpenAI-style SSE stream, so
 * tests exercise the actual parsing path end to end rather than just
 * asserting that functions were called with the right arguments.
 */
export function mockSseFetch(deltas: string[]): typeof fetch {
  const lines: string[] = [];
  deltas.forEach((delta, i) => {
    lines.push(
      `data: ${JSON.stringify({
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "test-model",
        choices: [
          {
            index: 0,
            delta: i === 0 ? { role: "assistant", content: delta } : { content: delta },
            finish_reason: null,
          },
        ],
      })}\n\n`,
    );
  });
  lines.push(
    `data: ${JSON.stringify({
      id: "chatcmpl-test",
      object: "chat.completion.chunk",
      created: 1700000000,
      model: "test-model",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    })}\n\n`,
  );
  lines.push("data: [DONE]\n\n");

  const body = lines.join("");

  return vi.fn(async () =>
    new Response(body, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    }),
  ) as unknown as typeof fetch;
}
