import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import ConnectionSettings from "./ConnectionSettings.svelte";

describe("ConnectionSettings", () => {
  it("pre-fills the form with the current values", async () => {
    const user = userEvent.setup();
    render(ConnectionSettings, {
      props: {
        baseURL: "http://localhost:8000/v1",
        model: "paeka-qwen",
        apiKey: "",
        onSave: vi.fn(),
      },
    });

    await user.click(screen.getByText("Connection"));
    expect(screen.getByPlaceholderText("http://localhost:8000/v1")).toHaveValue(
      "http://localhost:8000/v1",
    );
    expect(screen.getByPlaceholderText("paeka-qwen")).toHaveValue("paeka-qwen");
  });

  it("saves trimmed values and closes the panel", async () => {
    const onSave = vi.fn();
    const user = userEvent.setup();
    render(ConnectionSettings, {
      props: { baseURL: "", model: "", apiKey: "", onSave },
    });

    await user.click(screen.getByText("Connection"));
    await user.type(screen.getByPlaceholderText("http://localhost:8000/v1"), "  http://x/v1  ");
    await user.type(screen.getByPlaceholderText("paeka-qwen"), "  my-model  ");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(onSave).toHaveBeenCalledExactlyOnceWith({
      baseURL: "http://x/v1",
      model: "my-model",
      apiKey: "",
    });

    const details = document.querySelector("details");
    expect(details).not.toHaveAttribute("open");
  });
});
