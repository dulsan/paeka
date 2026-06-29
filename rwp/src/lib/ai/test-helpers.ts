import { vi } from "vitest";

interface MockReactReply {
  response: string;
  tool_calls?: Array<{ id: string; name: string; args: Record<string, unknown>; result: string; ok: boolean }>;
}

/**
 * Builds a fetch mock returning the actual JSON shape /api/agent/react
 * responds with, so tests exercise ReactAgentTransport's real parsing
 * path end to end rather than just asserting functions were called with
 * the right arguments.
 */
export function mockJsonFetch(reply: string | MockReactReply): typeof fetch {
  const body: MockReactReply = typeof reply === "string" ? { response: reply, tool_calls: [] } : reply;
  return vi.fn(async () =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  ) as unknown as typeof fetch;
}
