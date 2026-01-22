# Add `OpenaiRealtimeHandler` to `ToolDependencies`

## Summary

This PR exposes the `OpenaiRealtimeHandler` instance to tools via the `ToolDependencies` dataclass, enabling tools to interact with users during long-running or interactive `Tool.__call__()` executions.

## Changes

- **`core_tools.py`**: Added `openai_realtime_handler: OpenaiRealtimeHandler | None = None` field to `ToolDependencies`
- **`openai_realtime.py`**: Inject `self` into `deps.openai_realtime_handler` during handler initialization

## Motivation

Some tools require extended execution times or need to communicate with the user mid-execution (e.g., asking for clarification, providing progress updates, or streaming intermediate results). Without access to the `OpenaiRealtimeHandler`, tools were isolated and unable to leverage the realtime connection for user interaction.

By injecting the handler into the dependencies, tools can now:
- Send messages to the user during execution
- Access the realtime connection for bidirectional communication
- Provide a more interactive experience during complex operations

## Important Note

⚠️ **This PR should be paired with concurrent tool execution** to prevent blocking the `OpenaiRealtimeHandler` event loop. If tools run synchronously on the main handler thread, having access to the handler becomes ineffective since the handler itself would be blocked waiting for the tool to complete.

Concurrent/background tool execution ensures the realtime handler remains responsive while tools leverage it for user interaction.

## Testing

- Verified tool registration and initialization logs
- Confirmed `openai_realtime_handler` is properly injected and accessible from tools
