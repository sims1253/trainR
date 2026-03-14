/**
 * Pi Tool Trace Extension
 * 
 * Captures tool events and logs them to stdout for trace extraction.
 * Supports both event-based (0.54.x) and hook-based (0.55.x) APIs.
 */

const logEvent = (type, data) => {
  console.log(JSON.stringify({
    type,
    ...data,
    timestamp: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
  }));
};

// Debug: Log when extension file is loaded
logEvent("extension_loaded", { name: "pi-tool-trace-extension" });

/**
 * Event-based API (older Pi versions)
 */
export default function toolTraceExtension(pi) {
  logEvent("extension_initialized", { name: "pi-tool-trace-extension", api: "event" });
  
  pi.on("tool_execution_start", async (event) => {
    logEvent("tool_execution_start", {
      toolCallId: event.toolCallId,
      toolName: event.toolName,
      args: event.args || {},
    });
  });

  pi.on("tool_execution_end", async (event) => {
    logEvent("tool_execution_end", {
      toolCallId: event.toolCallId,
      toolName: event.toolName,
      isError: event.isError || false,
    });
  });

  pi.on("tool_call", async (event) => {
    logEvent("tool_call", {
      toolCallId: event.toolCallId,
      toolName: event.toolName,
      input: event.input,
    });
  });

  pi.on("tool_result", async (event) => {
    logEvent("tool_result", {
      toolCallId: event.toolCallId,
      toolName: event.toolName,
      isError: event.isError,
    });
  });
}

/**
 * Hook-based API (Pi 0.55.x+)
 */
export const name = "pi-tool-trace-extension";
export const hooks = {
  async onToolCall(ctx, event) {
    logEvent("tool_call", {
      toolCallId: event.toolCallId,
      toolName: event.toolName,
      input: event.input,
    });
  },
  async onToolResult(ctx, event) {
    logEvent("tool_result", {
      toolCallId: event.toolCallId,
      toolName: event.toolName,
      isError: event.isError,
    });
  },
  async onToolExecutionStart(ctx, event) {
    logEvent("tool_execution_start", {
      toolCallId: event.toolCallId,
      toolName: event.toolName,
      args: event.args || {},
    });
  },
  async onToolExecutionEnd(ctx, event) {
    logEvent("tool_execution_end", {
      toolCallId: event.toolCallId,
      toolName: event.toolName,
      isError: event.isError || false,
    });
  }
};
