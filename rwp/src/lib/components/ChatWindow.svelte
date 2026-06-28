<script lang="ts">
  import type { Chat } from "@ai-sdk/svelte";
  import { createAgentSession, type AgentSessionConfig } from "../ai/agent";
  import MessageList from "./MessageList.svelte";
  import ChatInput from "./ChatInput.svelte";
  import ConnectionSettings from "./ConnectionSettings.svelte";

  interface Props {
    initialConfig?: Partial<AgentSessionConfig>;
  }

  let { initialConfig = {} }: Props = $props();

  // Deliberately a one-time snapshot, not a live binding to
  // initialConfig -- these become independently editable via
  // ConnectionSettings afterward. Svelte's compiler notes this pattern
  // ("state_referenced_locally") since it's also a common mistake when
  // someone *did* want live reactivity; here it's intentional.
  let baseURL = $state(initialConfig.baseURL ?? "http://localhost:8000/v1");
  let model = $state(initialConfig.model ?? "paeka-qwen");
  let apiKey = $state(initialConfig.apiKey ?? "");

  let chat = $state<Chat | undefined>(undefined);
  let setupError: string | undefined = $state(undefined);

  // Recreates the session whenever connection settings change. This is
  // deliberate -- switching backend/model mid-conversation makes the
  // existing history's relevance ambiguous, so starting fresh is the
  // less surprising behaviour. Lives in $effect (not $derived) because
  // constructing a Chat session is a side effect, not a pure computation.
  $effect(() => {
    setupError = undefined;
    try {
      chat = createAgentSession({
        baseURL,
        model,
        apiKey: apiKey || undefined,
        fetch: initialConfig.fetch,
      });
    } catch (err) {
      setupError = err instanceof Error ? err.message : String(err);
      chat = undefined;
    }
  });

  function handleSend(text: string) {
    void chat?.sendMessage({ text });
  }

  function handleSaveSettings(values: { baseURL: string; model: string; apiKey: string }) {
    baseURL = values.baseURL;
    model = values.model;
    apiKey = values.apiKey;
  }

  const isBusy = $derived(chat?.status === "submitted" || chat?.status === "streaming");
</script>

<div class="chat-window">
  <header class="chat-window__header">
    <h1 class="chat-window__title">Research Assistant</h1>
    <ConnectionSettings {baseURL} {model} {apiKey} onSave={handleSaveSettings} />
  </header>

  {#if setupError}
    <p class="chat-window__error" role="alert">{setupError}</p>
  {:else if chat}
    <MessageList messages={chat.messages} status={chat.status} />
    {#if chat.error}
      <p class="chat-window__error" role="alert">{chat.error.message}</p>
    {/if}
    <ChatInput disabled={isBusy} onSend={handleSend} />
  {/if}
</div>

<style>
  .chat-window {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .chat-window__header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-4);
    padding: var(--space-4) var(--space-6);
    border-bottom: 1px solid var(--color-rule);
    background: var(--color-paper-raised);
  }

  .chat-window__title {
    font-family: var(--font-content);
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0;
    color: var(--color-ink);
  }

  .chat-window__error {
    font-family: var(--font-ui);
    font-size: 0.875rem;
    color: var(--color-error);
    background: var(--color-error-bg);
    max-width: 42rem;
    width: calc(100% - 2 * var(--space-4));
    margin: 0 auto var(--space-2);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius);
  }
</style>
