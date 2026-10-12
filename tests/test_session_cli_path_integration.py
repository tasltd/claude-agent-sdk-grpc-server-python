"""The SDK really starts the session's CLI binary (TAS-1071, S3).

``test_session_cli_path.py`` proves the right ``cli_path`` lands on the options.
This file proves the SDK then executes that file, with the sandbox id in its
environment, through the real ``ClaudeSDKClient`` and the CLI bundled with the
SDK. No network: the CLI has no credentials here, so a prompt comes back at once
as "Not logged in" without calling the API.

The stand-in for ``sbx-cli`` behaves like it: it records its argv and
environment, refuses to run without a well-formed ``TASCIM_SANDBOX_ID``
(exit 64), then execs the real CLI. Because it execs, the process the SDK
started IS the CLI, as it is for ``sbx-cli`` up to the ``sudo`` hop.
"""
from __future__ import annotations

import asyncio
import json
import os
import stat
import time
from pathlib import Path

import claude_agent_sdk
import pytest

from claude_agent_grpc_server.sdk.session_manager import (
    ALLOWED_CLI_PATHS_VAR,
    CLI_PATH_VAR,
    SANDBOX_ID_VAR,
    SessionConfig,
    SessionManager,
)

SBX_A = "0b5c3a8e-2f41-4d6a-9c1e-7a2b4c6d8e01"
SBX_B = "f00dfeed-1234-4abc-8def-0123456789ab"
BUNDLED_CLI = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"


def _wrapper(path: Path, log: Path, then: str) -> str:
    """A shell script that records each call to ``log``, then runs ``then``."""
    path.write_text(f"""#!/bin/sh
python3 - "$@" <<'PY' >> '{log}'
import json, os, sys
print(json.dumps({{"argv": sys.argv[1:], "pid": os.getppid(),
                  "sandbox_id": os.environ.get("TASCIM_SANDBOX_ID"),
                  "cli_path": os.environ.get("TASCIM_CLI_PATH")}}))
PY
id="${{TASCIM_SANDBOX_ID:-}}"
case "$id" in
  ''|*[!0-9a-f-]*) echo "stand-in sbx-cli: no sandbox id" >&2; exit 64 ;;
esac
{then}
""")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _calls(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
    except (FileNotFoundError, IndexError):
        return False
    return state != "Z"


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """No credentials, no network, and nothing written to the real home."""
    if not BUNDLED_CLI.is_file():
        pytest.fail(f"the SDK's bundled CLI is missing at {BUNDLED_CLI}; this test needs it")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    for key in list(os.environ):
        if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_OAUTH", "CLAUDE_CONFIG")):
            monkeypatch.delenv(key, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    return project


def _manager(monkeypatch, *paths: str) -> SessionManager:
    monkeypatch.setenv(ALLOWED_CLI_PATHS_VAR, ":".join(paths))
    return SessionManager()


def _config(project: Path, cli: str, sandbox_id: str) -> SessionConfig:
    return SessionConfig(
        project_path=str(project), model="sonnet", max_turns=1,
        environment={CLI_PATH_VAR: cli, SANDBOX_ID_VAR: sandbox_id},
    )


async def _turn(manager: SessionManager, session_id: str) -> list:
    async def run():
        return [m async for m in manager.send_prompt(session_id, "say hi")]
    return await asyncio.wait_for(run(), timeout=90)


async def test_the_sdk_starts_the_sessions_cli_with_its_sandbox_id(offline, tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    cli = _wrapper(tmp_path / "sbx-cli", log, f'exec "{BUNDLED_CLI}" "$@"')
    manager = _manager(monkeypatch, cli)

    info = await manager.create_session(_config(offline, cli, SBX_A))
    msgs = await _turn(manager, info.id)

    calls = _calls(log)
    session_calls = [c for c in calls if "--output-format" in c["argv"]]
    assert len(session_calls) == 1, calls
    assert session_calls[0]["sandbox_id"] == SBX_A
    assert session_calls[0]["cli_path"] == cli
    assert session_calls[0]["argv"][:2] == ["--output-format", "stream-json"]
    # The CLI ran and answered: the turn closed with its own reply.
    assert any("Not logged in" in (m.content or "") for m in msgs), [m.content for m in msgs]

    # Interrupt goes through the wrapper to the CLI and back (the control
    # request travels on the stdin/stdout the SDK opened on the wrapper).
    assert await manager.interrupt(info.id) is True

    # Cancel: deleting the session ends the process the SDK started.
    pid = session_calls[0]["pid"]
    assert _alive(pid)
    await manager.delete_session(info.id)
    for _ in range(50):
        if not _alive(pid):
            break
        await asyncio.sleep(0.1)
    assert not _alive(pid), f"the CLI (pid {pid}) outlived its session"


async def test_the_version_probe_has_no_sandbox_id(offline, tmp_path, monkeypatch):
    """The SDK runs ``<cli> -v`` before connecting, with the runner's environment only.

    sbx-cli refuses that call (no sandbox id). The SDK swallows the failure, so
    the session still starts. Pinned here so a change on either side is seen.
    """
    log = tmp_path / "calls.jsonl"
    cli = _wrapper(tmp_path / "sbx-cli", log, f'exec "{BUNDLED_CLI}" "$@"')
    monkeypatch.delenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", raising=False)
    manager = _manager(monkeypatch, cli)

    info = await manager.create_session(_config(offline, cli, SBX_A))
    msgs = await _turn(manager, info.id)
    await manager.delete_session(info.id)

    assert not [m for m in msgs if m.type == "error"], [m.content for m in msgs]
    probes = [c for c in _calls(log) if c["argv"] == ["-v"]]
    assert probes, "the SDK no longer probes the CLI version; revisit the runner note"
    assert all(c["sandbox_id"] is None for c in probes)


async def test_a_cli_that_exits_at_once_still_proves_it_was_started(offline, tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    cli = _wrapper(tmp_path / "sbx-cli", log, "exit 0")
    manager = _manager(monkeypatch, cli)

    info = await manager.create_session(_config(offline, cli, SBX_A))
    msgs = await _turn(manager, info.id)
    await manager.delete_session(info.id)

    assert [c["sandbox_id"] for c in _calls(log) if "--output-format" in c["argv"]] == [SBX_A]
    assert any(m.type == "error" for m in msgs), "a CLI that exits at once must surface an error"


async def test_a_cli_that_refuses_to_start_is_reported_in_seconds(offline, tmp_path, monkeypatch):
    """The sbx-exec refusal a person would hit, said in its own words, in seconds not a minute."""
    log = tmp_path / "calls.jsonl"
    cli = _wrapper(tmp_path / "sbx-cli", log,
                   'echo "sbx-exec: sandbox $TASCIM_SANDBOX_ID is not running" >&2; exit 69')
    manager = _manager(monkeypatch, cli)

    info = await manager.create_session(_config(offline, cli, SBX_A))
    start = time.monotonic()
    msgs = await _turn(manager, info.id)
    took = time.monotonic() - start
    await manager.delete_session(info.id)

    errors = [m.content for m in msgs if m.type == "error"]
    assert took < 10, f"the refusal took {took:.1f}s to surface"
    assert len(errors) == 1, errors
    assert "exit code 69" in errors[0]
    assert f"sandbox {SBX_A} is not running" in errors[0]


async def test_a_running_turn_stops_when_interrupted(offline, tmp_path, monkeypatch):
    """Plan risk R2: the turn is held open by an API that never answers, then interrupted."""
    reached = asyncio.Event()

    async def silent_api(reader, writer):
        reached.set()
        try:
            await reader.read(65536)
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(silent_api, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    log = tmp_path / "calls.jsonl"
    cli = _wrapper(tmp_path / "sbx-cli", log, f'exec "{BUNDLED_CLI}" "$@"')
    manager = _manager(monkeypatch, cli)
    config = _config(offline, cli, SBX_A)
    config.environment.update({"ANTHROPIC_API_KEY": "sk-ant-test-not-real",
                               "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"})
    info = await manager.create_session(config)

    turn = asyncio.ensure_future(_turn(manager, info.id))
    try:
        await asyncio.wait_for(reached.wait(), timeout=60)
        await asyncio.sleep(1.0)
        assert not turn.done(), "the turn ended before it could be interrupted"

        start = time.monotonic()
        assert await manager.interrupt(info.id) is True
        await asyncio.wait_for(turn, timeout=10)
        stopped_in = time.monotonic() - start
        assert stopped_in < 5, f"the turn took {stopped_in:.1f}s to stop"

        pid = next(c["pid"] for c in _calls(log) if "--output-format" in c["argv"])
        assert _alive(pid), "an interrupt stops the turn, not the session"
        await manager.delete_session(info.id)
        for _ in range(50):
            if not _alive(pid):
                break
            await asyncio.sleep(0.1)
        assert not _alive(pid)
    finally:
        if not turn.done():
            turn.cancel()
        server.close()


async def test_two_concurrent_sessions_start_their_own_cli(offline, tmp_path, monkeypatch):
    log_a, log_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    cli_a = _wrapper(tmp_path / "sbx-cli-a", log_a, f'exec "{BUNDLED_CLI}" "$@"')
    cli_b = _wrapper(tmp_path / "sbx-cli-b", log_b, f'exec "{BUNDLED_CLI}" "$@"')
    manager = _manager(monkeypatch, cli_a, cli_b)

    a = await manager.create_session(_config(offline, cli_a, SBX_A))
    b = await manager.create_session(_config(offline, cli_b, SBX_B))
    await asyncio.gather(_turn(manager, a.id), _turn(manager, b.id))
    await manager.delete_session(a.id)
    await manager.delete_session(b.id)

    started_a = [c for c in _calls(log_a) if "--output-format" in c["argv"]]
    started_b = [c for c in _calls(log_b) if "--output-format" in c["argv"]]
    assert [(c["cli_path"], c["sandbox_id"]) for c in started_a] == [(cli_a, SBX_A)]
    assert [(c["cli_path"], c["sandbox_id"]) for c in started_b] == [(cli_b, SBX_B)]
    assert started_a[0]["pid"] != started_b[0]["pid"]


async def test_a_refused_binary_is_never_executed(offline, tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    allowed = _wrapper(tmp_path / "sbx-cli", tmp_path / "allowed.jsonl", "exit 0")
    rogue = _wrapper(tmp_path / "rogue", log, "exit 0")
    manager = _manager(monkeypatch, allowed)

    with pytest.raises(ValueError):
        await manager.create_session(_config(offline, rogue, SBX_A))

    assert _calls(log) == []
