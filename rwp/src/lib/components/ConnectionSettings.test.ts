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
        apiKey: "",
        onSave: vi.fn(),
      },
    });

    await user.click(screen.getByText("Connection"));
    expect(screen.getByPlaceholderText("http://localhost:8000/v1")).toHaveValue(
      "http://localhost:8000/v1",
    );
  });

  it("saves trimmed values and closes the panel", async () => {
    const onSave = vi.fn();
    const user = userEvent.setup();
    render(ConnectionSettings, {
      props: { baseURL: "", apiKey: "", onSave },
    });

    await user.click(screen.getByText("Connection"));
    await user.type(screen.getByPlaceholderText("http://localhost:8000/v1"), "  http://x/v1  ");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(onSave).toHaveBeenCalledExactlyOnceWith({
      baseURL: "http://x/v1",
      apiKey: "",
    });

    const details = document.querySelector("details");
    expect(details).not.toHaveAttribute("open");
  });
});
