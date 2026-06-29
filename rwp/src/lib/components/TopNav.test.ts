import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import TopNav from "./TopNav.svelte";

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
});

describe("TopNav", () => {
  it("renders Chat as the active tab and Archive/Settings as disabled placeholders", () => {
    render(TopNav);
    expect(screen.getByRole("tab", { name: "Chat" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Archive" })).toBeDisabled();
    expect(screen.getByRole("tab", { name: "Settings" })).toBeDisabled();
  });

  it("ignores clicks on a disabled tab", async () => {
    const user = userEvent.setup();
    render(TopNav);
    await user.click(screen.getByRole("tab", { name: "Archive" }));
    expect(screen.getByRole("tab", { name: "Chat" })).toHaveAttribute("aria-selected", "true");
  });

  it("toggles and persists the theme", async () => {
    const user = userEvent.setup();
    render(TopNav);
    await user.click(screen.getByRole("button", { name: /switch to dark theme/i }));
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("rwp:theme")).toBe("dark");
    expect(screen.getByRole("button", { name: /switch to light theme/i })).toBeInTheDocument();
  });
});
