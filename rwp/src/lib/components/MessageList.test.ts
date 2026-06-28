import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/svelte";
import type { UIMessage, ChatStatus } from "ai";
import MessageList from "./MessageList.svelte";

function textMessage(id: string, role: "user" | "assistant", text: string): UIMessage {
  return { id, role, parts: [{ type: "text", text }] } as UIMessage;
}

describe("MessageList", () => {
  it("shows an empty-state prompt when there are no messages", () => {
    render(MessageList, { props: { messages: [], status: "ready" as ChatStatus } });
    expect(screen.getByText(/nothing here yet/i)).toBeInTheDocument();
  });

  it("renders messages in order", () => {
    const messages = [
      textMessage("1", "user", "first"),
      textMessage("2", "assistant", "second"),
      textMessage("3", "user", "third"),
    ];
    render(MessageList, { props: { messages, status: "ready" as ChatStatus } });

    const rendered = screen.getAllByRole("article").map((el) => el.textContent);
    expect(rendered[0]).toContain("first");
    expect(rendered[1]).toContain("second");
    expect(rendered[2]).toContain("third");
  });

  it("only marks the last assistant message as streaming while busy", () => {
    const messages = [
      textMessage("1", "user", "question"),
      textMessage("2", "assistant", "partial answer"),
    ];
    const { container } = render(MessageList, {
      props: { messages, status: "streaming" as ChatStatus },
    });
    const cursors = container.querySelectorAll(".message__cursor");
    expect(cursors).toHaveLength(1);
  });

  it("shows no streaming cursor once status is ready", () => {
    const messages = [
      textMessage("1", "user", "question"),
      textMessage("2", "assistant", "final answer"),
    ];
    const { container } = render(MessageList, {
      props: { messages, status: "ready" as ChatStatus },
    });
    expect(container.querySelectorAll(".message__cursor")).toHaveLength(0);
  });
});
