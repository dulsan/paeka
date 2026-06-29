const STORAGE_KEY = "rwp:theme";

export type Theme = "light" | "dark";

function isTheme(value: string | null): value is Theme {
  return value === "light" || value === "dark";
}

function systemPreference(): Theme {
  const matches =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  return matches ? "dark" : "light";
}

function readStored(): Theme | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return isTheme(raw) ? raw : null;
  } catch {
    // localStorage unavailable (private mode, disabled storage, etc.) --
    // fall back to the system default rather than throwing.
    return null;
  }
}

/** Resolves the theme that should be active: stored choice, else system default. */
export function getTheme(): Theme {
  return readStored() ?? systemPreference();
}

/** Applies a theme to the document and persists it as the user's explicit choice. */
export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Non-fatal: the theme still applies for this session, it just won't
    // be remembered next launch.
  }
}

/**
 * Call once on startup: resolves and applies the initial theme, returns it.
 *
 * Deliberately does NOT call applyTheme()/persist to localStorage here --
 * if the user never explicitly toggled, we want next launch to still
 * follow a changed system preference rather than being locked to
 * whatever we happened to resolve once and silently wrote down.
 */
export function initTheme(): Theme {
  const theme = getTheme();
  document.documentElement.dataset.theme = theme;
  return theme;
}

export function toggleTheme(current: Theme): Theme {
  const next: Theme = current === "dark" ? "light" : "dark";
  applyTheme(next);
  return next;
}
