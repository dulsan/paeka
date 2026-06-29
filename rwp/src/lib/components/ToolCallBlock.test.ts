import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/svelte";
import ToolCallBlock from "./ToolCallBlock.svelte";

describe("ToolCallBlock", () => {
  it("shows the tool name and a Done status with its output", () => {
    render(ToolCallBlock, {
      props: {
        part: {
          type: "dynamic-tool",
          toolCallId: "call_1",
          toolName: "calculator",
          state: "output-available",
          input: { expr: "2+2" },
          output: "4",
        },
      },
    });

    expect(screen.getByText("calculator")).toBeInTheDocument();
    expect(screen.getByText("Done")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
  });

  it("shows a Running status while only input is available", () => {
    render(ToolCallBlock, {
      props: {
        part: {
          type: "dynamic-tool",
          toolCallId: "call_1",
          toolName: "web_search",
          state: "input-available",
          input: { query: "paeka" },
        },
      },
    });

    expect(screen.getByText("Running")).toBeInTheDocument();
  });

  it("shows a Failed status and the error text, not an output block", () => {
    render(ToolCallBlock, {
      props: {
        part: {
          type: "dynamic-tool",
          toolCallId: "call_1",
          toolName: "web_search",
          state: "output-error",
          input: { query: "paeka" },
          errorText: "[MCP ERROR] timeout",
        },
      },
    });

    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("[MCP ERROR] timeout")).toBeInTheDocument();
    expect(screen.queryByText("Output")).not.toBeInTheDocument();
  });

  it("pretty-prints object input as JSON", () => {
    render(ToolCallBlock, {
      props: {
        part: {
          type: "dynamic-tool",
          toolCallId: "call_1",
          toolName: "calculator",
          state: "output-available",
          input: { expr: "2+2", precision: 2 },
          output: "4",
        },
      },
    });

    expect(screen.getByText(/"expr": "2\+2"/)).toBeInTheDocument();
  });
});
