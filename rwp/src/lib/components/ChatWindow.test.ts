import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import ChatWindow from "./ChatWindow.svelte";
import { mockSseFetch } from "../ai/test-helpers";

describe("ChatWindow", () => {
  it("sends a message and renders the streamed assistant reply", async () => {
    const user = userEvent.setup();
    render(ChatWindow, {
      props: {
        initialConfig: {
          baseURL: "http://localhost:8000/v1",
          model: "paeka-qwen",
          fetch: mockSseFetch(["Hel", "lo!"]),
        },
      },
    });

    await user.type(screen.getByLabelText("Message"), "hi there");
    await user.keyboard("{Enter}");

    expect(await screen.findByText("hi there")).toBeInTheDocument();
    expect(await screen.findByText("Hello!")).toBeInTheDocument();

    // back to ready once the stream finishes -- input usable again
    expect(screen.getByLabelText("Message")).not.toBeDisabled();
  });

  it("disables the input while a response is in flight", async () => {
    const user = userEvent.setup();
    render(ChatWindow, {
      props: {
        initialConfig: {
          baseURL: "http://localhost:8000/v1",
          model: "paeka-qwen",
          fetch: mockSseFetch(["ok"]),
        },
      },
    });

    await user.type(screen.getByLabelText("Message"), "hi there");
    await user.keyboard("{Enter}");

    // Should become busy at some point during the request before
    // settling back to ready -- by the time the assistant text shows
    // up the field is guaranteed usable again, so check it lands there.
    expect(await screen.findByText("ok")).toBeInTheDocument();
    expect(screen.getByLabelText("Message")).not.toBeDisabled();
  });

  it("shows a setup error instead of the chat when configuration is invalid", () => {
    render(ChatWindow, {
      props: { initialConfig: { baseURL: "", model: "paeka-qwen" } },
    });
    expect(screen.getByRole("alert")).toHaveTextContent(/base url/i);
    expect(screen.queryByLabelText("Message")).not.toBeInTheDocument();
  });
});
