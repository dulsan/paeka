<script lang="ts">
  import type { UIMessage } from "ai";
  import { renderMarkdown } from "../markdown";
  import ToolCallBlock from "./ToolCallBlock.svelte";

  interface Props {
    message: UIMessage;
    streaming?: boolean;
  }

  let { message, streaming = false }: Props = $props();

  const roleLabel = $derived(message.role === "user" ? "You" : "Assistant");

  // [SEAM] message.parts can interleave text and tool-call parts in
  // stream order (text, then a tool call, then more text). Adjacent text
  // parts are merged into one rendered block -- markdown-it needs a full
  // paragraph's text to parse correctly, and rendering "Hello" / " world"
  // as two separate <p> elements would visibly fragment a single sentence
  // that just happened to arrive as two stream chunks. Anything that
  // isn't text or a dynamic tool call (e.g. "step-start") is a structural
  // stream marker, not content, and is intentionally skipped here.
  type Segment =
    | { kind: "text"; text: string; isLast: boolean }
    | { kind: "tool"; part: Extract<UIMessage["parts"][number], { type: "dynamic-tool" }> };

  const segments = $derived.by((): Segment[] => {
    const result: Segment[] = [];
    for (const part of message.parts) {
      if (part.type === "text") {
        const last = result[result.length - 1];
        if (last && last.kind === "text") {
          last.text += part.text;
        } else {
          result.push({ kind: "text", text: part.text, isLast: false });
        }
      } else if (part.type === "dynamic-tool") {
        result.push({ kind: "tool", part });
      }
    }
    for (let i = result.length - 1; i >= 0; i--) {
      const seg = result[i];
      if (seg.kind === "text") {
        seg.isLast = true;
        break;
      }
    }
    return result;
  });
</script>

<article class="message message--{message.role}" aria-label="{roleLabel} message">
  <div class="message__rule" aria-hidden="true"></div>
  <div class="message__body">
    <p class="message__label">{roleLabel}</p>
    {#each segments as segment, index (index)}
      {#if segment.kind === "text"}
        <div class="message__text">
          {@html renderMarkdown(segment.text)}{#if streaming && segment.isLast}<span
              class="message__cursor"
              aria-hidden="true"
            ></span>{/if}
        </div>
      {:else}
        <ToolCallBlock part={segment.part} />
      {/if}
    {/each}
    {#if segments.length === 0 && streaming}
      <div class="message__text"><span class="message__cursor" aria-hidden="true"></span></div>
    {/if}
  </div>
</article>

<style>
  .message {
    display: flex;
    gap: var(--space-3);
    padding: var(--space-4) 0;
  }

  .message__rule {
    flex: 0 0 3px;
    border-radius: var(--radius);
    align-self: stretch;
  }

  .message--user .message__rule {
    background: var(--color-accent-user);
  }

  .message--assistant .message__rule {
    background: var(--color-accent-assistant);
  }

  .message__body {
    flex: 1;
    min-width: 0;
  }

  .message__label {
    margin: 0 0 var(--space-1);
    font-family: var(--font-ui);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .message--user .message__label {
    color: var(--color-accent-user);
  }

  .message--assistant .message__label {
    color: var(--color-accent-assistant);
  }

  .message__text {
    font-family: var(--font-content);
    font-size: 1.0625rem;
    line-height: 1.6;
    color: var(--color-ink);
    word-break: break-word;
  }

  .message__cursor {
    display: inline-block;
    width: 2px;
    height: 1em;
    margin-left: 2px;
    background: var(--color-accent-assistant);
    vertical-align: text-bottom;
    animation: blink 1s step-start infinite;
  }

  @keyframes blink {
    50% {
      opacity: 0;
    }
  }

  /* Rendered-markdown content arrives via {@html}, so Svelte's scoped
     classes never reach it -- everything below has to be :global(). */
  .message__text :global(p) {
    margin: 0 0 var(--space-3);
  }
  .message__text :global(p:last-child) {
    margin-bottom: 0;
  }
  .message__text :global(ul),
  .message__text :global(ol) {
    margin: 0 0 var(--space-3);
    padding-left: 1.4em;
  }
  .message__text :global(li) {
    margin: 0 0 var(--space-1);
  }
  .message__text :global(blockquote) {
    margin: 0 0 var(--space-3);
    padding-left: var(--space-3);
    border-left: 2px solid var(--color-rule);
    color: var(--color-ink-soft);
  }
  .message__text :global(a) {
    color: var(--color-accent-user);
    text-decoration: underline;
    text-decoration-color: var(--color-rule);
  }
  .message__text :global(hr) {
    border: none;
    border-top: 1px solid var(--color-rule);
    margin: var(--space-4) 0;
  }
  .message__text :global(code) {
    font-family: var(--font-mono);
    font-size: 0.875em;
    background: var(--color-code-bg);
    padding: 0.1em 0.35em;
    border-radius: var(--radius);
  }
  .message__text :global(pre) {
    margin: 0 0 var(--space-3);
    padding: var(--space-3);
    background: var(--color-code-bg);
    border-radius: var(--radius);
    overflow-x: auto;
  }
  .message__text :global(pre code) {
    background: none;
    padding: 0;
    color: var(--color-ink);
    font-size: 0.875rem;
    line-height: 1.5;
  }
  .message__text :global(.hljs-keyword),
  .message__text :global(.hljs-literal) {
    color: var(--color-code-keyword);
  }
  .message__text :global(.hljs-string) {
    color: var(--color-code-string);
  }
  .message__text :global(.hljs-comment) {
    color: var(--color-code-comment);
    font-style: italic;
  }
  .message__text :global(.hljs-number) {
    color: var(--color-code-number);
  }
  .message__text :global(.hljs-title),
  .message__text :global(.hljs-title.function_) {
    font-weight: 600;
  }
  .message__text :global(.hljs-attr),
  .message__text :global(.hljs-attribute) {
    color: var(--color-ink-soft);
  }
  /* KaTeX ships its own layout-critical CSS (imported globally in main.ts);
     this is only a type-family/size nudge so equations sit visually at
     home in a serif reading column instead of looking pasted in. */
  .message__text :global(.katex) {
    font-size: 1.05em;
  }
  .message__text :global(.katex-display) {
    margin: var(--space-3) 0;
    overflow-x: auto;
  }
</style>
