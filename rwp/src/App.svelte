<script lang="ts">
  import "./app.css";
  import TopNav from "./lib/components/TopNav.svelte";
  import ChatWindow from "./lib/components/ChatWindow.svelte";
</script>

<div class="app-shell">
  <div class="app-shell__nav">
    <TopNav />
  </div>
  <div class="app-shell__content">
    <ChatWindow />
  </div>
</div>

<style>
  .app-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .app-shell__nav {
    flex: 0 0 auto;
    padding: var(--space-3) var(--space-6) 0;
    border-bottom: 1px solid var(--color-rule);
    background: var(--color-paper-raised);
  }

  .app-shell__content {
    flex: 1;
    min-height: 0;
  }
</style>
