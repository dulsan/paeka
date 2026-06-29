import { describe, it, expect } from "vitest";
import { render } from "@testing-library/svelte";
import App from "./App.svelte";

describe("App (production mount diagnostic)", () => {
  it("mounts with zero props, exactly as main.ts renders it, without throwing", () => {
    expect(() => render(App)).not.toThrow();
  });

  it("renders TopNav and the chat header once mounted", () => {
    const { getByText } = render(App);
    expect(getByText("Chat")).toBeInTheDocument();
    expect(getByText("Research Assistant")).toBeInTheDocument();
  });
});
