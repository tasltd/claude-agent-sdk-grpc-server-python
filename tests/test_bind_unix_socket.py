"""A grpc server must be able to bind a Unix domain socket (S3/TAS-846).

Moving Tier 2's gRPC off TCP removes its dependency on the shared L2 bridge,
DHCP and four restored capabilities. grpcio binds ``unix:///path`` natively on
the server side. This asserts the bind actually produces the socket file (the
client side uses the same ``unix://`` target in ``aio.insecure_channel``).
"""
import asyncio
import os
from pathlib import Path

import pytest

from claude_agent_grpc_server.server import GRPCServer

SOCK = "/tmp/tascim-test-grpc.sock"


@pytest.mark.asyncio
async def test_grpc_binds_unix_socket():
    Path(SOCK).unlink(missing_ok=True)
    server = GRPCServer(address=f"unix://{SOCK}")
    await server.start()
    try:
        assert Path(SOCK).exists(), "server did not create the unix socket file"
    finally:
        await server.stop()
        Path(SOCK).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_explicit_address_wins_over_host_port():
    """When an address is supplied it is the canonical bind target."""
    Path(SOCK).unlink(missing_ok=True)
    server = GRPCServer(host="1.2.3.4", port=1, address=f"unix://{SOCK}")
    await server.start()
    try:
        assert Path(SOCK).exists(), "explicit unix address was not honoured"
    finally:
        await server.stop()
        Path(SOCK).unlink(missing_ok=True)
