import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// [NOTE] Plain Vite + Svelte, deliberately not SvelteKit -- the eventual
// Tauri shell wraps a static frontend bundle (no Node server at runtime),
// and per the SAD's data-flow diagram, AI orchestration is meant to live
// in the Rust layer, not in a server route. See src/lib/ai/agent.ts for
// where that seam actually is today (DirectChatTransport, swappable
// later for a Tauri-IPC transport with zero changes to the components).
export default defineConfig({
  plugins: [svelte()],
  // [FIX] Without this, Vitest (which runs in Node) resolves Svelte's
  // package exports to the server-side runtime build, which doesn't
  // implement mount() at all ("lifecycle_function_unavailable") --
  // jsdom provides DOM globals but doesn't change which export
  // condition Node resolves to. Forcing 'browser' here only affects
  // the test run; the normal dev/build pipeline already resolves
  // correctly since Vite is serving to an actual browser there.
  resolve: process.env.VITEST ? { conditions: ["browser"] } : undefined,
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});
