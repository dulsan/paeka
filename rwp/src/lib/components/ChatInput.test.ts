import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import ChatInput from "./ChatInput.svelte";

describe("ChatInput", () => {
  it("sends the trimmed text and clears the field on Enter", async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(ChatInput, { props: { onSend } });

    const field = screen.getByLabelText("Message");
    await user.type(field, "  hello there  ");
    await user.keyboard("{Enter}");

    expect(onSend).toHaveBeenCalledExactlyOnceWith("hello there");
    expect(field).toHaveValue("");
  });

  it("does not send on Shift+Enter, allowing a newline instead", async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(ChatInput, { props: { onSend } });

    const field = screen.getByLabelText("Message");
    await user.type(field, "line one");
    await user.keyboard("{Shift>}{Enter}{/Shift}");
    await user.type(field, "line two");

    expect(onSend).not.toHaveBeenCalled();
    expect(field).toHaveValue("line one\nline two");
  });

  it("does not send empty or whitespace-only input", async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(ChatInput, { props: { onSend } });

    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).not.toHaveBeenCalled();

    const field = screen.getByLabelText("Message");
    await user.type(field, "   ");
    await user.keyboard("{Enter}");
    expect(onSend).not.toHaveBeenCalled();
  });

  it("disables the field and button, and blocks sending, while disabled", async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(ChatInput, { props: { onSend, disabled: true } });

    expect(screen.getByLabelText("Message")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();

    // Enter shouldn't do anything even if somehow triggered on a disabled field
    await user.keyboard("{Enter}");
    expect(onSend).not.toHaveBeenCalled();
  });
});
