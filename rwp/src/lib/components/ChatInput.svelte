<script lang="ts">
  interface Props {
    disabled?: boolean;
    onSend: (text: string) => void;
  }

  let { disabled = false, onSend }: Props = $props();

  let value = $state("");

  function submit() {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    value = "";
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }
</script>

<form
  class="chat-input"
  onsubmit={(event) => {
    event.preventDefault();
    submit();
  }}
>
  <textarea
    class="chat-input__field"
    placeholder="Ask something... (Enter to send, Shift+Enter for a new line)"
    rows="2"
    bind:value
    onkeydown={handleKeydown}
    {disabled}
    aria-label="Message"
  ></textarea>
  <button type="submit" class="chat-input__send" disabled={disabled || !value.trim()}>
    Send
  </button>
</form>

<style>
  .chat-input {
    display: flex;
    gap: var(--space-3);
    align-items: flex-end;
    max-width: 42rem;
    width: 100%;
    margin: 0 auto;
    padding: var(--space-4);
    border-top: 1px solid var(--color-rule);
    background: var(--color-paper-raised);
  }

  .chat-input__field {
    flex: 1;
    font-family: var(--font-content);
    font-size: 1rem;
    line-height: 1.5;
    color: var(--color-ink);
    background: var(--color-paper-raised);
    border: 1px solid var(--color-rule);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    resize: vertical;
    min-height: 2.75rem;
  }

  .chat-input__field:disabled {
    color: var(--color-ink-soft);
    background: var(--color-paper);
  }

  .chat-input__send {
    font-family: var(--font-ui);
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--color-paper-raised);
    background: var(--color-accent-user);
    border: none;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-4);
    height: 2.75rem;
    cursor: pointer;
  }

  .chat-input__send:disabled {
    background: var(--color-rule);
    color: var(--color-ink-soft);
    cursor: not-allowed;
  }
</style>
