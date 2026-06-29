<script lang="ts">
  interface SettingsValues {
    baseURL: string;
    apiKey: string;
  }

  interface Props extends SettingsValues {
    onSave: (values: SettingsValues) => void;
  }

  let { baseURL, apiKey, onSave }: Props = $props();

  let open = $state(false);
  // One-time snapshot for editing, not a live binding -- see the same
  // note in ChatWindow.svelte. Drafts only flow back via onSave.
  let draftBaseURL = $state(baseURL);
  let draftApiKey = $state(apiKey);

  function handleSubmit(event: SubmitEvent) {
    event.preventDefault();
    onSave({
      baseURL: draftBaseURL.trim(),
      apiKey: draftApiKey.trim(),
    });
    open = false;
  }
</script>

<details class="settings" bind:open>
  <summary class="settings__summary">Connection</summary>
  <form class="settings__form" onsubmit={handleSubmit}>
    <label class="settings__field">
      <span>Base URL</span>
      <input type="text" bind:value={draftBaseURL} placeholder="http://localhost:8000/v1" />
    </label>
    <label class="settings__field">
      <span>API key (optional)</span>
      <input
        type="password"
        bind:value={draftApiKey}
        placeholder="leave blank if not required"
      />
    </label>
    <button type="submit">Save</button>
  </form>
</details>

<style>
  .settings {
    font-family: var(--font-ui);
    font-size: 0.875rem;
  }

  .settings__summary {
    cursor: pointer;
    color: var(--color-ink-soft);
    user-select: none;
  }

  .settings__form {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    margin-top: var(--space-2);
    padding: var(--space-3);
    background: var(--color-paper-raised);
    border: 1px solid var(--color-rule);
    border-radius: var(--radius);
    min-width: 18rem;
  }

  .settings__field {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  .settings__field span {
    font-size: 0.75rem;
    color: var(--color-ink-soft);
  }

  .settings__field input {
    font-family: var(--font-mono);
    font-size: 0.8125rem;
    padding: var(--space-2);
    border: 1px solid var(--color-rule);
    border-radius: var(--radius);
    background: var(--color-paper);
    color: var(--color-ink);
  }

  .settings__form button {
    align-self: flex-end;
    font-family: var(--font-ui);
    font-weight: 600;
    font-size: 0.8125rem;
    padding: var(--space-1) var(--space-3);
    background: var(--color-accent-user);
    color: var(--color-paper-raised);
    border: none;
    border-radius: var(--radius);
    cursor: pointer;
  }
</style>
