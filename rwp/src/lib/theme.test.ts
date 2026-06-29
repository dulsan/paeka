import { describe, it, expect, beforeEach } from "vitest";
import { getTheme, applyTheme, initTheme, toggleTheme } from "./theme";

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
});

describe("theme", () => {
  it("defaults to light with nothing stored and no system preference available", () => {
    expect(getTheme()).toBe("light");
  });

  it("respects a previously stored preference", () => {
    applyTheme("dark");
    expect(getTheme()).toBe("dark");
  });

  it("ignores a corrupted stored value rather than throwing", () => {
    localStorage.setItem("rwp:theme", "purple");
    expect(getTheme()).toBe("light");
  });

  it("initTheme applies the resolved theme to the document element", () => {
    localStorage.setItem("rwp:theme", "dark");
    const theme = initTheme();
    expect(theme).toBe("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("initTheme does not write a system-derived default to storage", () => {
    initTheme();
    expect(localStorage.getItem("rwp:theme")).toBeNull();
  });

  it("toggleTheme flips the theme and persists the explicit choice", () => {
    const next = toggleTheme("light");
    expect(next).toBe("dark");
    expect(localStorage.getItem("rwp:theme")).toBe("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
  });
});
