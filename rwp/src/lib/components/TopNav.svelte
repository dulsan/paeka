<script lang="ts">
  import { onMount } from "svelte";
  import { initTheme, toggleTheme, type Theme } from "../theme";

  interface NavTab {
    id: string;
    label: string;
    enabled: boolean;
  }

  // Only "Chat" does anything today. Archive (Paper Library/Lab Book) and
  // Settings are shown -- matching the LiLi reference's tab row and the
  // SRS's workspace navigation -- but kept disabled rather than wired to
  // nothing, so the affordance is honest about what currently exists.
  const tabs: NavTab[] = [
    { id: "chat", label: "Chat", enabled: true },
    { id: "archive", label: "Archive", enabled: false },
    { id: "settings", label: "Settings", enabled: false },
  ];

  let activeTab = $state("chat");
  let theme: Theme = $state("light");

  onMount(() => {
    theme = initTheme();
  });

  function selectTab(tab: NavTab) {
    if (!tab.enabled) return;
    activeTab = tab.id;
  }

  function handleToggleTheme() {
    theme = toggleTheme(theme);
  }
</script>

<nav class="top-nav" aria-label="Workspace sections">
  <div class="top-nav__tabs" role="tablist">
    {#each tabs as tab (tab.id)}
      <button
        type="button"
        class="top-nav__tab"
        class:top-nav__tab--active={activeTab === tab.id}
        role="tab"
        aria-selected={activeTab === tab.id}
        aria-disabled={!tab.enabled}
        disabled={!tab.enabled}
        title={tab.enabled ? undefined : "Coming soon"}
        onclick={() => selectTab(tab)}
      >
        {tab.label}
      </button>
    {/each}
  </div>

  <button
    type="button"
    class="top-nav__theme-toggle"
    onclick={handleToggleTheme}
    aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
    title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
  >
    {#if theme === "dark"}
      <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
        <circle cx="10" cy="10" r="4.5" fill="currentColor" />
        <g stroke="currentColor" stroke-width="1.4" stroke-linecap="round">
          <line x1="10" y1="1.5" x2="10" y2="3.5" />
          <line x1="10" y1="16.5" x2="10" y2="18.5" />
          <line x1="1.5" y1="10" x2="3.5" y2="10" />
          <line x1="16.5" y1="10" x2="18.5" y2="10" />
          <line x1="4.2" y1="4.2" x2="5.6" y2="5.6" />
          <line x1="14.4" y1="14.4" x2="15.8" y2="15.8" />
          <line x1="4.2" y1="15.8" x2="5.6" y2="14.4" />
          <line x1="14.4" y1="5.6" x2="15.8" y2="4.2" />
        </g>
      </svg>
    {:else}
      <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
        <path fill="currentColor" d="M14.5 11.4A6 6 0 0 1 8.6 5.5a6 6 0 1 0 5.9 5.9Z" />
      </svg>
    {/if}
  </button>
</nav>

<style>
  .top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
  }

  .top-nav__tabs {
    display: flex;
    gap: var(--space-4);
  }

  .top-nav__tab {
    font-family: var(--font-ui);
    font-size: 0.8125rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--color-ink-soft);
    background: none;
    border: none;
    padding: var(--space-1) 0 var(--space-2);
    border-bottom: 2px solid transparent;
    cursor: pointer;
  }

  .top-nav__tab:disabled {
    cursor: default;
    opacity: 0.45;
  }

  .top-nav__tab--active {
    color: var(--color-ink);
    border-bottom-color: var(--color-accent-user);
  }

  .top-nav__theme-toggle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.75rem;
    height: 1.75rem;
    flex: 0 0 auto;
    color: var(--color-ink-soft);
    background: none;
    border: 1px solid var(--color-rule);
    border-radius: var(--radius);
    cursor: pointer;
  }

  .top-nav__theme-toggle:hover {
    color: var(--color-ink);
    border-color: var(--color-ink-soft);
  }
</style>
