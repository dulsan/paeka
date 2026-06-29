<script lang="ts">
  interface DynamicToolPart {
    type: "dynamic-tool";
    toolName: string;
    toolCallId: string;
    state: "input-streaming" | "input-available" | "output-available" | "output-error" | string;
    input?: unknown;
    output?: unknown;
    errorText?: string;
  }

  interface Props {
    part: DynamicToolPart;
  }

  let { part }: Props = $props();

  const statusLabel = $derived(
    part.state === "output-error"
      ? "Failed"
      : part.state === "output-available"
        ? "Done"
        : "Running",
  );

  function formatValue(value: unknown): string {
    if (value === undefined) return "";
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }
</script>

<details class="tool-call" data-state={part.state}>
  <summary class="tool-call__summary">
    <span class="tool-call__name">{part.toolName}</span>
    <span class="tool-call__status">{statusLabel}</span>
  </summary>
  <div class="tool-call__body">
    {#if part.input !== undefined}
      <p class="tool-call__label">Input</p>
      <pre class="tool-call__pre">{formatValue(part.input)}</pre>
    {/if}
    {#if part.state === "output-error"}
      <p class="tool-call__label">Error</p>
      <pre class="tool-call__pre tool-call__pre--error">{part.errorText}</pre>
    {:else if part.output !== undefined}
      <p class="tool-call__label">Output</p>
      <pre class="tool-call__pre">{formatValue(part.output)}</pre>
    {/if}
  </div>
</details>

<style>
  .tool-call {
    margin: var(--space-2) 0;
    border: 1px solid var(--color-rule);
    border-radius: var(--radius);
    background: var(--color-paper-raised);
  }

  .tool-call__summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    cursor: pointer;
    font-family: var(--font-ui);
    font-size: 0.8125rem;
    list-style: none;
  }

  .tool-call__summary::-webkit-details-marker {
    display: none;
  }

  .tool-call__name {
    font-family: var(--font-mono);
    font-weight: 600;
    color: var(--color-accent-assistant);
  }

  .tool-call__name::before {
    content: "tool · ";
    font-family: var(--font-ui);
    font-weight: 400;
    color: var(--color-ink-soft);
  }

  .tool-call__status {
    margin-left: auto;
    color: var(--color-ink-soft);
    text-transform: uppercase;
    font-size: 0.6875rem;
    letter-spacing: 0.04em;
  }

  [data-state="output-error"] .tool-call__status {
    color: var(--color-error);
  }

  .tool-call__body {
    padding: 0 var(--space-3) var(--space-3);
    border-top: 1px solid var(--color-rule);
  }

  .tool-call__label {
    margin: var(--space-2) 0 var(--space-1);
    font-family: var(--font-ui);
    font-size: 0.6875rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-ink-soft);
  }

  .tool-call__pre {
    margin: 0;
    padding: var(--space-2);
    background: var(--color-paper);
    border-radius: var(--radius);
    font-family: var(--font-mono);
    font-size: 0.8125rem;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 16rem;
    overflow-y: auto;
  }

  .tool-call__pre--error {
    background: var(--color-error-bg);
    color: var(--color-error);
  }
</style>
