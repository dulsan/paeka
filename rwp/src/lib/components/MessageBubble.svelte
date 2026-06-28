<script lang="ts">
  import type { UIMessage } from "ai";

  interface Props {
    message: UIMessage;
    streaming?: boolean;
  }

  let { message, streaming = false }: Props = $props();

  const text = $derived(
    message.parts
      .filter((p): p is { type: "text"; text: string } => p.type === "text")
      .map((p) => p.text)
      .join(""),
  );

  const roleLabel = $derived(message.role === "user" ? "You" : "Assistant");
</script>

<article class="message message--{message.role}" aria-label="{roleLabel} message">
  <div class="message__rule" aria-hidden="true"></div>
  <div class="message__body">
    <p class="message__label">{roleLabel}</p>
    <div class="message__text">
      {text}{#if streaming}<span class="message__cursor" aria-hidden="true"></span>{/if}
    </div>
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
    white-space: pre-wrap;
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
</style>
