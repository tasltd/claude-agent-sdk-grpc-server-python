"""A CLI that exits while starting is reported in seconds, in its own words (TAS-1071).

The SDK waits at least 60 s for the CLI's first reply and does not notice that
the process has exited. A sandboxed session's CLI is sbx-cli, which exits at once
when it refuses (no sandbox id, a sudo rule missing, the sandbox not running), so
before this the person saw a silent session for a minute. ``connect_or_explain``
watches the process while the SDK starts and fails as soon as it exits.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from types import SimpleNamespace

import pytest

from claude_agent_grpc_server.sdk import session_manager as sm
from claude_agent_grpc_server.sdk.session_manager import CLIStartError, connect_or_explain


class Starting:
    """A client whose connect never finishes on its own, like the SDK's for a dead CLI."""

    def __init__(self, exit_after=None, code=64, finish_after=None, raise_after=None):
        self._transport = SimpleNamespace(_process=SimpleNamespace(returncode=None))
        self.exit_after, self.code = exit_after, code
        self.finish_after, self.raise_after = finish_after, raise_after
        self.cancelled = False

    async def connect(self):
        start = time.monotonic()
        try:
            while True:
                waited = time.monotonic() - start
                if self.exit_after is not None and waited >= self.exit_after:
                    self._transport._process.returncode = self.code
                if self.finish_after is not None and waited >= self.finish_after:
                    return
                if self.raise_after is not None and waited >= self.raise_after:
                    raise RuntimeError("initialize failed")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def test_a_cli_that_exits_is_reported_within_a_second():
    client = Starting(exit_after=0.1, code=64)
    tail = deque(["sbx-cli: TASCIM_SANDBOX_ID is missing or malformed\n"])

    start = time.monotonic()
    with pytest.raises(CLIStartError) as failed:
        await connect_or_explain(client, tail, poll_seconds=0.05)

    assert time.monotonic() - start < 1.0
    assert "exit code 64" in str(failed.value)
    assert "TASCIM_SANDBOX_ID is missing or malformed" in str(failed.value)
    assert client.cancelled, "the SDK's connect must be cancelled so it closes the process"


async def test_no_stderr_still_names_the_exit_code():
    with pytest.raises(CLIStartError) as failed:
        await connect_or_explain(Starting(exit_after=0, code=1), deque(), poll_seconds=0.01)
    assert str(failed.value) == "The agent program stopped while starting (exit code 1)."


async def test_the_last_non_blank_line_is_used_and_cut():
    tail = deque(["first\n", "x" * 5000 + "\n", "   \n"], maxlen=sm.STDERR_TAIL_LINES)
    with pytest.raises(CLIStartError) as failed:
        await connect_or_explain(Starting(exit_after=0), tail, poll_seconds=0.01)
    assert "first" not in str(failed.value)
    assert len(str(failed.value)) < 400


async def test_a_good_start_is_untouched():
    client = Starting(finish_after=0.15)
    await connect_or_explain(client, deque(), poll_seconds=0.01)
    assert not client.cancelled


async def test_a_connect_error_is_passed_through():
    with pytest.raises(RuntimeError, match="initialize failed"):
        await connect_or_explain(Starting(raise_after=0.05), deque(), poll_seconds=0.01)


async def test_a_client_with_no_transport_yet_is_waited_for():
    client = Starting(finish_after=0.1)
    del client._transport
    await connect_or_explain(client, deque(), poll_seconds=0.01)


async def test_cancelling_the_caller_cancels_the_connect():
    client = Starting()
    task = asyncio.ensure_future(connect_or_explain(client, deque(), poll_seconds=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.cancelled


def test_a_long_stderr_keeps_only_the_tail(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    manager = sm.SessionManager()
    sink = manager._stderr_sink("s1")
    for i in range(10_000):
        sink(f"line {i}\n")
    tail = manager._stderr_tails["s1"]
    assert len(tail) == sm.STDERR_TAIL_LINES
    assert tail[-1] == "line 9999\n"


async def test_deleting_a_session_drops_its_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    manager = sm.SessionManager()
    info = await manager.create_session(sm.SessionConfig(project_path="/tmp"))
    manager._stderr_sink(info.id)("boom\n")
    await manager.delete_session(info.id)
    assert info.id not in manager._stderr_tails


async def test_an_sdk_error_after_the_cli_exited_carries_the_cli_reason():
    class DiesLoudly(Starting):
        async def connect(self):
            await asyncio.sleep(0.05)
            self._transport._process.returncode = 69
            self._transport._process = None      # the SDK's own cleanup runs first
            raise RuntimeError("Command failed with exit code 69. Check stderr output for details")

    tail = deque(["sbx-exec: sandbox is not running\n"])
    with pytest.raises(CLIStartError) as failed:
        await connect_or_explain(DiesLoudly(), tail, poll_seconds=0.01)
    assert "exit code 69" in str(failed.value)
    assert "sandbox is not running" in str(failed.value)


async def test_stderr_that_lands_just_after_the_exit_is_still_used():
    tail: deque = deque()

    async def late():
        await asyncio.sleep(0.1)
        tail.append("late reason\n")

    asyncio.ensure_future(late())
    with pytest.raises(CLIStartError) as failed:
        await connect_or_explain(Starting(exit_after=0), tail, poll_seconds=0.01)
    assert "late reason" in str(failed.value)


async def test_the_sdk_errors_own_exit_code_is_used_if_the_process_was_never_seen():
    class Fast:
        async def connect(self):
            err = RuntimeError("Command failed")
            err.exit_code = 70
            raise err

    with pytest.raises(CLIStartError) as failed:
        await connect_or_explain(Fast(), deque(["why\n"]), poll_seconds=0.01)
    assert "exit code 70" in str(failed.value) and "why" in str(failed.value)
