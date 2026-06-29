import MarkdownIt from "markdown-it";
import * as markdownItKatexModule from "@vscode/markdown-it-katex";
import hljs from "highlight.js";
import DOMPurify from "dompurify";

// [SEAM] @vscode/markdown-it-katex ships as `exports.default = fn` (a
// transpiled-CJS shape). Different tools resolve that to different,
// confirmed-empirically-different depths of wrapping (Vitest: the bare
// function; plain Node ESM: one level of .default; Vite's production
// rolldown bundle: two levels of .default -- traced directly in the
// built output). A first attempt resolved this with a `while` loop at
// module scope, but the bundler's minifier collapsed that loop down to
// a single `typeof x === "function" ? x : x.default` check, silently
// handling only one of the two levels this package actually needs.
//
// The fix: wrap the plugin in a real function instead of resolving it
// to a value at module scope. markdown-it's `.use()` calls
// `plugin.apply(...)`, so as long as what we hand it is *syntactically*
// a function -- which markdownItKatexPlugin always is -- that call can
// never fail regardless of what minification did to anything inside
// it. The actual .default-walking happens lazily, at call time, against
// a closed-over variable -- ordinary property access a minifier has no
// reason to "optimize" away, unlike a chain of module-scope consts.
function markdownItKatexPlugin(md: MarkdownIt, options?: Record<string, unknown>): void {
  let plugin: unknown = markdownItKatexModule;
  for (let depth = 0; depth < 5 && typeof plugin !== "function"; depth++) {
    if (plugin && typeof plugin === "object" && "default" in plugin) {
      plugin = (plugin as { default: unknown }).default;
    } else {
      break;
    }
  }
  if (typeof plugin !== "function") {
    throw new Error(
      "markdown.ts: could not resolve @vscode/markdown-it-katex's plugin function from its module export.",
    );
  }
  (plugin as (md: MarkdownIt, options?: Record<string, unknown>) => void)(md, options);
}

// [SEAM] Single shared parser instance -- markdown-it instances are cheap
// to reuse across renders and expensive-ish to construct (loads its full
// rule set each time), so this module creates exactly one.
const md: MarkdownIt = new MarkdownIt({
  html: false, // raw HTML in model output is never executed as markup -- see sanitizeHtml() below, which is the actual security boundary, but no sense even parsing it as HTML here.
  linkify: true,
  breaks: true,
  highlight(code: string, lang: string): string {
    const language = lang && hljs.getLanguage(lang) ? lang : undefined;
    try {
      const result = language
        ? hljs.highlight(code, { language })
        : hljs.highlightAuto(code);
      return result.value;
    } catch {
      return md.utils.escapeHtml(code);
    }
  },
});

md.use(markdownItKatexPlugin, {
  throwOnError: false,
  // Bare \begin{...}\end{...} blocks are common in LLM output that's
  // half-remembering LaTeX conventions -- worth rendering even without
  // the $$ delimiters a strict reading of the spec would require.
  enableBareBlocks: true,
});

// Links (including auto-linked URLs from linkify) open in a real new tab
// rather than navigating the chat window away -- there's no "back"
// affordance in this app shell, and that matters more once it's running
// inside a Tauri webview instead of a regular browser tab.
const defaultLinkOpen =
  md.renderer.rules.link_open ??
  ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));
md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
  tokens[idx].attrSet("target", "_blank");
  tokens[idx].attrSet("rel", "noopener noreferrer");
  return defaultLinkOpen(tokens, idx, options, env, self);
};

/**
 * Renders LLM-authored markdown (including LaTeX via KaTeX and fenced code
 * via highlight.js) to sanitized HTML safe to inject with `{@html}`.
 *
 * The sanitize step is not optional: this text originates from a model
 * whose own output can be steered by retrieved documents, tool results, or
 * web search content (prompt injection), so it is untrusted by construction
 * regardless of how much we trust the model weights themselves.
 */
export function renderMarkdown(source: string): string {
  const rawHtml = md.render(source);
  return sanitizeHtml(rawHtml);
}

function sanitizeHtml(rawHtml: string): string {
  return DOMPurify.sanitize(rawHtml, {
    // KaTeX's output mixes plain HTML with inline MathML and (for things
    // like \sqrt radicals and \overline) SVG paths -- all three profiles
    // are needed or KaTeX's own markup gets silently stripped.
    USE_PROFILES: { html: true, svg: true, svgFilters: true, mathMl: true },
    ADD_ATTR: ["target", "rel"], // linkify-generated/forced rel+target on anchors; harmless to keep, needed for noopener.
  });
}
