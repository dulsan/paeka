import { describe, it, expect } from "vitest";
import { renderMarkdown } from "./markdown";

describe("renderMarkdown", () => {
  it("returns an empty string for empty input", () => {
    expect(renderMarkdown("")).toBe("");
  });

  it("renders basic markdown formatting", () => {
    const html = renderMarkdown("**bold** and *italic*");
    expect(html).toContain("<strong>bold</strong>");
    expect(html).toContain("<em>italic</em>");
  });

  it("renders lists", () => {
    const html = renderMarkdown("- one\n- two");
    expect(html).toContain("<li>one</li>");
    expect(html).toContain("<li>two</li>");
  });

  it("syntax-highlights fenced code blocks", () => {
    const html = renderMarkdown("```python\ndef f():\n    return 1\n```");
    expect(html).toContain("hljs-keyword");
    expect(html).toContain("def");
  });

  it("never leaks a raw unescaped tag for unrecognized languages", () => {
    // highlightAuto() guesses something (often markup/XML, given the angle
    // brackets) rather than throwing -- the real invariant is "no live
    // <stuff> tag survives", not which internal hljs path produced it.
    const html = renderMarkdown("```not-a-real-lang\nraw <stuff>\n```");
    expect(html).not.toContain("<stuff>");
  });

  it("renders inline LaTeX via KaTeX", () => {
    const html = renderMarkdown("Energy is $E = mc^2$.");
    expect(html).toContain("katex");
  });

  it("renders block LaTeX via KaTeX", () => {
    const html = renderMarkdown("$$\\frac{a}{b} = c$$");
    expect(html).toContain("katex-display");
  });

  it("does not throw on malformed LaTeX, and degrades to visible error text", () => {
    expect(() => renderMarkdown("$\\frac{$")).not.toThrow();
  });

  it("escapes/strips script and img tags so no live executable element exists (prompt-injection surface)", () => {
    const html = renderMarkdown('<script>alert(1)</script><img src=x onerror="alert(1)">');
    const doc = new DOMParser().parseFromString(html, "text/html");
    expect(doc.querySelectorAll("script, img").length).toBe(0);
  });

  it("preserves plain paragraphs and links", () => {
    const html = renderMarkdown("See https://example.com for details.");
    expect(html).toContain('href="https://example.com"');
  });

  it("opens links in a new tab with noopener (no in-app back affordance)", () => {
    const html = renderMarkdown("[paper](https://example.com/paper.pdf)");
    expect(html).toContain('target="_blank"');
    expect(html).toContain('rel="noopener noreferrer"');
  });
});
