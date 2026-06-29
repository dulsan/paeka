#!/usr/bin/env node
/**
 * scripts/smoke-test-bundle.mjs
 * ===============================
 * Executes the real built dist/assets/*.js bundle under jsdom and
 * asserts it actually mounts content into #app.
 *
 * Why this exists, not just `vitest run` and `vite build`:
 * - `vite build` only checks that the bundle is syntactically valid and
 *   that all imports resolve to *something*. It does not execute any of
 *   the resulting code, so it cannot catch a value being the wrong
 *   *shape* at runtime.
 * - `vitest run` executes real code, but through Vitest's own
 *   Node-based module resolution -- which can (and, for at least one
 *   real dependency in this app, did) resolve a CJS package's default
 *   export to a different shape than Vite's production rolldown
 *   bundler does for the exact same import statement. Every test was
 *   green while the real built app threw on load.
 *
 * This script closes that gap by actually running the same JS a
 * browser would load, just with jsdom standing in for the DOM. It is
 * intentionally not a full browser (no real layout/paint), but it
 * executes every line of top-level module code and the initial mount,
 * which is exactly where this category of bug lives.
 *
 * Usage: bun run build && node scripts/smoke-test-bundle.mjs
 */

import { JSDOM } from "jsdom";
import { readdirSync } from "fs";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";

const rwpRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const assetsDir = path.join(rwpRoot, "dist", "assets");

const jsFile = readdirSync(assetsDir).find((f) => f.endsWith(".js"));
if (!jsFile) {
  console.error(`No built .js bundle found in ${assetsDir}. Run "bun run build" first.`);
  process.exit(1);
}

const dom = new JSDOM('<!doctype html><html><body><div id="app"></div></body></html>', {
  url: "http://localhost/",
  pretendToBeVisual: true,
});

// Only the globals real-world dependencies in this app have turned out
// to touch at module-load/initial-mount time. If a future dependency
// needs another one, the error message naming it is the signal to add
// it here -- that's a sign this script is doing its job, not that it's
// broken.
const GLOBALS_TO_BRIDGE = [
  "window", "document", "navigator", "localStorage",
  "HTMLElement", "Element", "Node", "Text", "Comment",
  "MutationObserver", "EventTarget", "DocumentFragment", "SVGElement",
  "HTMLMediaElement", "Event", "CustomEvent", "KeyboardEvent", "MouseEvent",
];
for (const key of GLOBALS_TO_BRIDGE) {
  if (dom.window[key] !== undefined) {
    Object.defineProperty(global, key, { value: dom.window[key], configurable: true });
  }
}
global.requestAnimationFrame = (cb) => setTimeout(cb, 0);
global.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
global.window.matchMedia = global.matchMedia;

let failed = false;

try {
  await import(pathToFileURL(path.join(assetsDir, jsFile)).href);
  // Mounting involves a microtask hop or two (Svelte effects flush
  // async); give it a moment before inspecting the result.
  await new Promise((resolve) => setTimeout(resolve, 300));

  const app = document.getElementById("app");
  const html = app?.innerHTML ?? "";

  check("the bundle executed without throwing", true);
  check("#app has mounted content (not blank)", html.length > 0);
  check('TopNav rendered ("Chat" tab present)', html.includes(">Chat<"));
  check('ChatWindow rendered ("Research Assistant" header present)', html.includes("Research Assistant"));
} catch (err) {
  failed = true;
  console.error("✗ Bundle threw during execution:\n");
  console.error(err.stack || err);
}

function check(label, ok) {
  if (ok) {
    console.log(`✓ ${label}`);
  } else {
    failed = true;
    console.log(`✗ ${label}`);
  }
}

if (failed) {
  console.error("\nSmoke test FAILED -- the production bundle does not mount correctly.");
  process.exit(1);
} else {
  console.log("\nSmoke test passed.");
}
