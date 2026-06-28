<script lang="ts">
  import type { UIMessage, ChatStatus } from "ai";
  import MessageBubble from "./MessageBubble.svelte";

  interface Props {
    messages: UIMessage[];
    status: ChatStatus;
  }

  let { messages, status }: Props = $props();

  let scrollEl: HTMLDivElement | undefined = $state();

  $effect(() => {
    // Re-runs whenever messages (length or content) changes.
    void messages;
    if (scrollEl) {
      scrollEl.scrollTop = scrollEl.scrollHeight;
    }
  });

  function isStreamingAt(index: number): boolean {
    return (
      (status === "streaming" || status === "submitted") &&
      index === messages.length - 1 &&
      messages[index]?.role === "assistant"
    );
  }
</script>

<div
  class="message-list"
  bind:this={scrollEl}
  role="log"
  aria-live="polite"
  aria-label="Conversation"
>
  {#if messages.length === 0}
    <p class="message-list__empty">Nothing here yet -- ask something to get started.</p>
  {:else}
    {#each messages as message, index (message.id)}
      <MessageBubble {message} streaming={isStreamingAt(index)} />
    {/each}
  {/if}
</div>

<style>
  .message-list {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-6) var(--space-4);
    max-width: 42rem;
    width: 100%;
    margin: 0 auto;
  }

  .message-list__empty {
    font-family: var(--font-ui);
    color: var(--color-ink-soft);
    font-size: 0.9375rem;
    margin-top: var(--space-8);
    text-align: center;
  }
</style>
