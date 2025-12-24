"""
Claude Agent gRPC Server Entry Point.

This module provides the main entry point for starting the gRPC server.
"""

import asyncio
import grpc
from grpc import aio
import os
import logging
import signal
from typing import Optional

from .proto import claude_agent_pb2_grpc as pb2_grpc
from .services.claude_service import ClaudeAgentServicer
from .sdk.session_manager import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GRPCServer:
    """
    gRPC Server wrapper for Claude Agent service.

    Provides clean startup/shutdown and signal handling.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 10,
    ):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.server: Optional[aio.Server] = None
        self.session_manager = SessionManager()
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the gRPC server."""
        self.server = aio.server(
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
            ]
        )

        # Add the Claude Agent service
        servicer = ClaudeAgentServicer(self.session_manager)
        pb2_grpc.add_ClaudeAgentServiceServicer_to_server(servicer, self.server)

        # Bind to address
        # Use explicit format - gRPC can have issues with address resolution in containers
        if self.host in ("0.0.0.0", ""):
            # Bind to all interfaces - try both IPv4 and IPv6
            address = f"[::]:{self.port}"
        else:
            address = f"{self.host}:{self.port}"

        port_result = self.server.add_insecure_port(address)
        if port_result == 0:
            # If IPv6 failed, try IPv4 only
            if address.startswith("[::]:"):
                logger.warning(f"Failed to bind to {address}, trying IPv4 only")
                address = f"0.0.0.0:{self.port}"
                port_result = self.server.add_insecure_port(address)

            if port_result == 0:
                raise RuntimeError(f"Failed to bind to {address}")

        logger.info(f"Starting Claude Agent gRPC server on {address}")
        await self.server.start()
        logger.info("Server started successfully")

    async def stop(self, grace_period: float = 5.0) -> None:
        """Stop the gRPC server gracefully."""
        if self.server:
            logger.info("Shutting down server...")
            await self.server.stop(grace_period)
            logger.info("Server stopped")

    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self.server:
            await self.server.wait_for_termination()

    async def run(self) -> None:
        """Run the server until interrupted."""
        await self.start()

        # Set up signal handlers
        loop = asyncio.get_running_loop()

        def signal_handler():
            logger.info("Received shutdown signal")
            self._shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        try:
            # Wait for either termination or shutdown signal
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(self.server.wait_for_termination()),
                    asyncio.create_task(self._shutdown_event.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        finally:
            await self.stop()


async def serve(
    host: str = "0.0.0.0",
    port: int = 50051,
) -> None:
    """
    Start the gRPC server.

    Args:
        host: Host address to bind to (default: 0.0.0.0)
        port: Port to listen on (default: 50051)
    """
    server = GRPCServer(host=host, port=port)
    await server.run()


def main() -> None:
    """Main entry point for the server."""
    # Get configuration from environment
    host = os.getenv("GRPC_HOST", "0.0.0.0")
    port = int(os.getenv("GRPC_PORT", "50051"))

    logger.info(f"Claude Agent gRPC Server v0.1.0")
    logger.info(f"Configuration: host={host}, port={port}")

    try:
        asyncio.run(serve(host=host, port=port))
    except KeyboardInterrupt:
        logger.info("Server interrupted")


if __name__ == "__main__":
    main()
