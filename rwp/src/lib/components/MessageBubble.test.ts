import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/svelte";
import type { UIMessage } from "ai";
import MessageBubble from "./MessageBubble.svelte";

function textMessage(role: "user" | "assistant", text: string): UIMessage {
  return {
    id: "m1",
    role,
    parts: [{ type: "text", text }],
  } as UIMessage;
}

describe("MessageBubble", () => {
  it("renders user messages with a 'You' label", () => {
    render(MessageBubble, { props: { message: textMessage("user", "Hello there") } });
    expect(screen.getByText("You")).toBeInTheDocument();
    expect(screen.getByText("Hello there")).toBeInTheDocument();
  });

  it("renders assistant messages with an 'Assistant' label", () => {
    render(MessageBubble, { props: { message: textMessage("assistant", "Hi back") } });
    expect(screen.getByText("Assistant")).toBeInTheDocument();
    expect(screen.getByText("Hi back")).toBeInTheDocument();
  });

  it("joins multiple text parts into one block", () => {
    const message = {
      id: "m2",
      role: "assistant",
      parts: [
        { type: "text", text: "Hello" },
        { type: "text", text: " world" },
      ],
    } as UIMessage;
    render(MessageBubble, { props: { message } });
    expect(screen.getByText("Hello world")).toBeInTheDocument();
  });

  it("shows a streaming cursor only when streaming is true", () => {
    const { container, rerender } = render(MessageBubble, {
      props: { message: textMessage("assistant", "thinking"), streaming: true },
    });
    expect(container.querySelector(".message__cursor")).not.toBeNull();

    rerender({ message: textMessage("assistant", "thinking"), streaming: false });
    expect(container.querySelector(".message__cursor")).toBeNull();
  });

  it("ignores non-text parts rather than crashing", () => {
    const message = {
      id: "m3",
      role: "assistant",
      parts: [{ type: "step-start" }, { type: "text", text: "still here" }],
    } as unknown as UIMessage;
    render(MessageBubble, { props: { message } });
    expect(screen.getByText("still here")).toBeInTheDocument();
  });
});
