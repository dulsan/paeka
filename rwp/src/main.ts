import "katex/dist/katex.min.css";
import { mount } from "svelte";
import App from "./App.svelte";

const target = document.getElementById("app");
if (!target) {
  throw new Error("#app element not found -- check index.html");
}

const app = mount(App, { target });

export default app;
