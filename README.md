# Claude Agent gRPC Server

gRPC server wrapping the Claude Agent SDK for network-accessible Claude sessions.

## Installation

```bash
pip install claude-agent-grpc-server
```

Or from source:

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run the server
claude-grpc-server

# Or with custom port
GRPC_PORT=50052 claude-grpc-server
```

## Features

- **Full Claude Agent SDK integration** with streaming responses
- **Ephemeral session credentials** - per-session API keys/OAuth tokens with auto-cleanup
- **Multiple credential sources** - settings files, environment variables, mounted `.claude` folders
- **Dangerous command blocking** - security hooks to prevent destructive operations
- **AskUserQuestion support** - interactive sessions with user prompts
- **Session persistence** - output logging and state recovery

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_HOST` | `0.0.0.0` | Host to bind to |
| `GRPC_PORT` | `50051` | Port to listen on |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `CLAUDE_SETTINGS_PATH` | - | Path to settings file |

## Proto Regeneration

If you modify the proto file, regenerate with:

```bash
python -m grpc_tools.protoc -I../proto \
  --python_out=src/claude_agent_grpc_server/proto \
  --grpc_python_out=src/claude_agent_grpc_server/proto \
  ../proto/claude_agent.proto
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
```

## Part of claude-agent-grpc-tools

This package is part of the [claude-agent-grpc-tools](https://github.com/tasltd/claude-agent-grpc-tools) monorepo.

## License

MIT
