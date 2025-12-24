"""
Claude Agent gRPC Server

A gRPC server that wraps the official Claude Agent SDK,
enabling network-accessible Claude sessions for containers
and microservices.
"""

__version__ = "0.1.0"

from .services.claude_service import ClaudeAgentServicer
from .sdk.session_manager import SessionManager, SessionConfig, SessionInfo, StreamMessage
from .server import serve, main, GRPCServer

__all__ = [
    "ClaudeAgentServicer",
    "SessionManager",
    "SessionConfig",
    "SessionInfo",
    "StreamMessage",
    "GRPCServer",
    "serve",
    "main",
    "__version__",
]
