"""A session may name its CLI binary, but only one the runner allows (TAS-1071, S3).

One runner per host serves every sandboxed session on it. The host sends
``TASCIM_CLI_PATH`` (the binary to start, ``sbx-cli``) and ``TASCIM_SANDBOX_ID``
in each session's environment. A path from a session is untrusted input, so the
runner accepts it only if it is on the runner's own allowlist
(``TASCIM_ALLOWED_CLI_PATHS``), and only with a sandbox id in the exact form
``sbx-cli`` accepts.

These tests read the options off the real ``send_prompt`` code path, like
``test_session_env_reaches_cli.py``. ``test_session_cli_path_integration.py``
proves the SDK then really starts that file.
"""
from __future__ import annotations

import asyncio
import json
import os
import stat

import grpc
import pytest
from claude_agent_sdk.types import ClaudeAgentOptions
from test_options_are_sdk_legal import _build_command
from test_send_prompt_turn_boundary import StubSDKClient, _collect, _install_session, _turn

from claude_agent_grpc_server.proto import claude_agent_pb2 as pb2
from claude_agent_grpc_server.sdk import session_manager as sm
from claude_agent_grpc_server.sdk.session_manager import (
    ALLOWED_CLI_PATHS_VAR,
    CLI_PATH_VAR,
    DEFAULT_ALLOWED_CLI_PATHS,
    SANDBOX_ID_VAR,
    SessionConfig,
    SessionManager,
    SessionRefusedError,
    allowed_cli_paths,
    session_cli_path,
)
from claude_agent_grpc_server.services.claude_service import ClaudeAgentServicer

SBX_A = "0b5c3a8e-2f41-4d6a-9c1e-7a2b4c6d8e01"
SBX_B = "f00dfeed-1234-4abc-8def-0123456789ab"


def _executable(path) -> str:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture
def cli(tmp_path):
    return _executable(tmp_path / "sbx-cli")


@pytest.fixture
def cli_b(tmp_path):
    return _executable(tmp_path / "sbx-cli-b")


def _config(env=None, project_path="/tmp") -> SessionConfig:
    return SessionConfig(project_path=project_path, model="claude-opus-5", environment=dict(env or {}))


@pytest.fixture
def make_manager(tmp_path, monkeypatch):
    """A real SessionManager whose allowlist is whatever the test sets first."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    def build(allowlist=None):
        if allowlist is None:
            monkeypatch.delenv(ALLOWED_CLI_PATHS_VAR, raising=False)
        else:
            monkeypatch.setenv(ALLOWED_CLI_PATHS_VAR, allowlist)
        return SessionManager()

    return build


@pytest.fixture
def recorded_options(monkeypatch):
    seen: list[ClaudeAgentOptions] = []
    real = sm.ClaudeAgentOptions

    def recorder(**kwargs):
        built = real(**kwargs)
        seen.append(built)
        return built

    monkeypatch.setattr(sm, "ClaudeAgentOptions", recorder)
    return seen


async def _prompt(manager, session_id: str, env) -> list:
    client = StubSDKClient()
    info = _install_session(manager, client, session_id=session_id)
    manager._sessions[session_id] = (info, _config(env), client)
    client.script_turn(_turn())
    return await _collect(manager.send_prompt(session_id, "hi"))


def _errors(msgs) -> list[str]:
    return [m.content for m in msgs if m.is_error]


# --- the allowlist --------------------------------------------------------


def test_the_allowlist_defaults_to_sbx_cli():
    assert allowed_cli_paths({}) == frozenset(DEFAULT_ALLOWED_CLI_PATHS)
    assert DEFAULT_ALLOWED_CLI_PATHS == ("/usr/libexec/tascim/sbx-cli",)


def test_an_empty_allowlist_allows_nothing():
    assert allowed_cli_paths({ALLOWED_CLI_PATHS_VAR: ""}) == frozenset()
    assert allowed_cli_paths({ALLOWED_CLI_PATHS_VAR: ":::"}) == frozenset()


def test_the_allowlist_splits_on_colons_and_drops_relative_entries():
    raw = "/opt/a::bin/b:/opt/c:./d:/opt/a"
    assert allowed_cli_paths({ALLOWED_CLI_PATHS_VAR: raw}) == {"/opt/a", "/opt/c"}


def test_a_thousand_entries_are_all_kept():
    raw = ":".join(f"/opt/cli-{i}" for i in range(1000))
    assert len(allowed_cli_paths({ALLOWED_CLI_PATHS_VAR: raw})) == 1000


def test_the_manager_reads_the_allowlist_once(make_manager, monkeypatch, cli):
    manager = make_manager("")
    monkeypatch.setenv(ALLOWED_CLI_PATHS_VAR, cli)

    with pytest.raises(SessionRefusedError) as refused:
        session_cli_path(_config({CLI_PATH_VAR: cli, SANDBOX_ID_VAR: SBX_A}), manager._allowed_cli_paths)
    assert refused.value.code == "denied"


# --- one session's request ------------------------------------------------


@pytest.mark.parametrize("env", [{}, {CLI_PATH_VAR: ""}, {CLI_PATH_VAR: "", SANDBOX_ID_VAR: ""}],
                         ids=["none", "empty-path", "both-empty"])
def test_no_cli_path_means_the_runners_own_cli(env):
    assert session_cli_path(_config(env), frozenset()) is None


def test_a_good_pair_is_accepted(cli):
    assert session_cli_path(_config({CLI_PATH_VAR: cli, SANDBOX_ID_VAR: SBX_A}), frozenset({cli})) == cli


def test_a_path_outside_the_allowlist_is_denied(cli, tmp_path):
    other = _executable(tmp_path / "bash")
    with pytest.raises(SessionRefusedError) as refused:
        session_cli_path(_config({CLI_PATH_VAR: other, SANDBOX_ID_VAR: SBX_A}), frozenset({cli}))
    assert refused.value.code == "denied"
    assert "not allowed on this runner" in str(refused.value)
    assert cli in str(refused.value)


def test_an_empty_allowlist_denies_every_path(cli):
    with pytest.raises(SessionRefusedError) as refused:
        session_cli_path(_config({CLI_PATH_VAR: cli, SANDBOX_ID_VAR: SBX_A}), frozenset())
    assert refused.value.code == "denied"
    assert "Allowed: none" in str(refused.value)


@pytest.mark.parametrize("path_of", [
    lambda cli: cli + "/",                 # a trailing slash is not the same path
    lambda cli: cli.replace("/sbx-cli", "/./sbx-cli"),
    lambda cli: cli.replace("/sbx-cli", "/x/../sbx-cli"),
    lambda cli: "sbx-cli",                 # relative
], ids=["trailing-slash", "dot", "dotdot", "relative"])
def test_only_the_exact_allowed_string_matches(cli, path_of):
    with pytest.raises(SessionRefusedError):
        session_cli_path(_config({CLI_PATH_VAR: path_of(cli), SANDBOX_ID_VAR: SBX_A}), frozenset({cli}))


@pytest.mark.parametrize("sandbox_id", [None, ""], ids=["missing", "empty"])
def test_a_cli_path_needs_a_sandbox_id(cli, sandbox_id):
    env = {CLI_PATH_VAR: cli}
    if sandbox_id is not None:
        env[SANDBOX_ID_VAR] = sandbox_id
    with pytest.raises(SessionRefusedError) as refused:
        session_cli_path(_config(env), frozenset({cli}))
    assert refused.value.code == "invalid"
    assert SANDBOX_ID_VAR in str(refused.value)


@pytest.mark.parametrize("sandbox_id", [
    SBX_A.upper(),                         # sbx-cli takes lowercase only
    SBX_A.replace("-", ""),
    SBX_A[:-1],
    SBX_A + "0",
    SBX_A + "\n",
    "\n" + SBX_A,
    " " + SBX_A,
    SBX_A[:8] + "-zzzz" + SBX_A[13:],
    "../../etc/passwd",
    "x" * 100_000,
], ids=["upper", "no-dashes", "short", "long", "trailing-newline", "leading-newline",
        "leading-space", "non-hex", "traversal", "huge"])
def test_a_malformed_sandbox_id_is_refused(cli, sandbox_id):
    with pytest.raises(SessionRefusedError) as refused:
        session_cli_path(_config({CLI_PATH_VAR: cli, SANDBOX_ID_VAR: sandbox_id}), frozenset({cli}))
    assert refused.value.code == "invalid"
    assert len(str(refused.value)) < 400, "a huge value must not be echoed whole"


def test_a_sandbox_id_without_a_cli_path_is_refused():
    """The host asked for a sandbox. Running the runner's own CLI instead would not be one."""
    with pytest.raises(SessionRefusedError) as refused:
        session_cli_path(_config({SANDBOX_ID_VAR: SBX_A}), frozenset(DEFAULT_ALLOWED_CLI_PATHS))
    assert "outside its sandbox" in str(refused.value)


@pytest.mark.parametrize("bad", [
    "/usr/libexec/tascim/sbx-cli\n",
    "/usr/libexec/tascim/sbx-cli\n/bin/sh",
    "/usr/libexec/tascim/sbx-cli\x00",
    "/usr/libexec/tascim/\x1b[2Jsbx-cli",
    "/" + "a" * 5000,
    "/" + "a/" * 50_000,
], ids=["trailing-newline", "embedded-newline", "nul", "escape", "over-path-max", "huge"])
def test_a_long_or_control_character_path_is_refused(bad):
    allowed = frozenset({bad, "/usr/libexec/tascim/sbx-cli"})   # even if an operator listed it
    with pytest.raises(SessionRefusedError) as refused:
        session_cli_path(_config({CLI_PATH_VAR: bad, SANDBOX_ID_VAR: SBX_A}), allowed)
    assert refused.value.code == "invalid"
    assert len(str(refused.value)) < 400


def test_an_allowed_path_that_is_not_executable_is_refused(tmp_path):
    plain = tmp_path / "sbx-cli"
    plain.write_text("not a program")
    for path in (str(plain), str(tmp_path), str(tmp_path / "missing")):
        with pytest.raises(SessionRefusedError) as refused:
            session_cli_path(_config({CLI_PATH_VAR: path, SANDBOX_ID_VAR: SBX_A}), frozenset({path}))
        assert "not an executable file" in str(refused.value)


# --- create, prompt, summary, restore -------------------------------------


async def test_create_session_refuses_and_leaves_nothing_behind(make_manager, cli, tmp_path):
    manager = make_manager(cli)
    other = _executable(tmp_path / "other")

    with pytest.raises(SessionRefusedError):
        await manager.create_session(_config({CLI_PATH_VAR: other, SANDBOX_ID_VAR: SBX_A}))

    assert manager._sessions == {}
    assert manager._ephemeral_credentials == {}
    assert manager._question_responses == {}


async def test_create_session_accepts_a_good_pair(make_manager, cli):
    manager = make_manager(cli)
    info = await manager.create_session(_config({CLI_PATH_VAR: cli, SANDBOX_ID_VAR: SBX_A}))
    assert info.id in manager._sessions


async def test_with_no_variables_the_options_are_unchanged(make_manager, recorded_options):
    manager = make_manager(None)
    msgs = await _prompt(manager, "s1", {})

    assert not _errors(msgs)
    assert recorded_options[-1].cli_path is None
    assert recorded_options[-1].env == {}
    _build_command(recorded_options[-1])


async def test_the_cli_path_and_sandbox_id_reach_the_options(make_manager, recorded_options, cli):
    manager = make_manager(cli)
    msgs = await _prompt(manager, "s1", {CLI_PATH_VAR: cli, SANDBOX_ID_VAR: SBX_A, "OTHER": "kept"})

    assert not _errors(msgs)
    options = recorded_options[-1]
    assert options.cli_path == cli
    assert options.env[SANDBOX_ID_VAR] == SBX_A
    assert options.env["OTHER"] == "kept"
    _build_command(options)


async def test_a_refused_path_fails_the_prompt_too(make_manager, recorded_options, cli, tmp_path):
    """A session stored before the allowlist shrank cannot start a binary it no longer may."""
    manager = make_manager(cli)
    other = _executable(tmp_path / "other")

    msgs = await _prompt(manager, "s1", {CLI_PATH_VAR: other, SANDBOX_ID_VAR: SBX_A})

    assert any("not allowed on this runner" in e for e in _errors(msgs))
    assert recorded_options == []


async def test_two_concurrent_sessions_keep_their_own_path_and_id(make_manager, recorded_options, cli, cli_b):
    manager = make_manager(f"{cli}:{cli_b}")

    results = await asyncio.gather(
        _prompt(manager, "a", {CLI_PATH_VAR: cli, SANDBOX_ID_VAR: SBX_A}),
        _prompt(manager, "b", {CLI_PATH_VAR: cli_b, SANDBOX_ID_VAR: SBX_B}),
        _prompt(manager, "c", {}),
    )

    assert not [e for msgs in results for e in _errors(msgs)]
    seen = {(o.cli_path, o.env.get(SANDBOX_ID_VAR)) for o in recorded_options}
    assert seen == {(cli, SBX_A), (cli_b, SBX_B), (None, None)}


async def test_a_hundred_sessions_do_not_bleed(make_manager, recorded_options, tmp_path):
    paths = [_executable(tmp_path / f"cli-{i}") for i in range(4)]
    manager = make_manager(":".join(paths))
    wanted = {
        f"s{i}": (paths[i % 4], f"{i:08x}-0000-4000-8000-{i:012x}")
        for i in range(100)
    }

    await asyncio.gather(*[
        _prompt(manager, sid, {CLI_PATH_VAR: path, SANDBOX_ID_VAR: sbx})
        for sid, (path, sbx) in wanted.items()
    ])

    got = sorted((o.cli_path, o.env[SANDBOX_ID_VAR]) for o in recorded_options)
    assert got == sorted(wanted.values())


async def test_the_summary_runs_through_the_sessions_cli(make_manager, monkeypatch, cli):
    manager = make_manager(cli)
    built: list[ClaudeAgentOptions] = []

    class SummaryClient:
        def __init__(self, options):
            built.append(options)

        async def connect(self):
            return None

        async def query(self, _prompt):
            return None

        async def disconnect(self):
            return None

    monkeypatch.setattr(sm, "ClaudeSDKClient", SummaryClient)
    client = StubSDKClient()
    info = _install_session(manager, client)
    manager._sessions["s1"] = (info, _config({CLI_PATH_VAR: cli, SANDBOX_ID_VAR: SBX_A}), client)
    manager._session_outputs["s1"] = [
        sm.SessionOutput(timestamp="2026-10-12T00:00:00Z", message_type="text", content="did a thing " * 20)
    ]

    await manager.generate_summary("s1")

    assert built and built[-1].cli_path == cli
    assert built[-1].env[SANDBOX_ID_VAR] == SBX_A


def _save_state(tmp_path, session_id: str, environment: dict) -> None:
    state_dir = tmp_path / ".claude" / "state" / "orchestration" / "grpc_sessions"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{session_id}.json").write_text(json.dumps({
        "session": {"id": session_id, "name": session_id, "project_path": "/tmp",
                    "model": "sonnet", "created_at": "2026-10-12T00:00:00+00:00"},
        "config": {"project_path": "/tmp", "environment": environment},
    }))


async def test_restore_checks_todays_allowlist(make_manager, tmp_path, cli):
    manager = make_manager("")              # the path was allowed once, not now
    _save_state(tmp_path, "old", {CLI_PATH_VAR: cli, SANDBOX_ID_VAR: SBX_A})

    assert await manager.restore_session("old") is None
    assert "old" not in manager._sessions


async def test_restore_keeps_an_allowed_session(make_manager, tmp_path, cli):
    manager = make_manager(cli)
    _save_state(tmp_path, "ok", {CLI_PATH_VAR: cli, SANDBOX_ID_VAR: SBX_A})

    assert await manager.restore_session("ok") is not None


# --- the gRPC status a refusal maps to ------------------------------------


class _AbortedError(Exception):
    pass


class _Context:
    def __init__(self):
        self.code = None
        self.details = None

    async def abort(self, code, details):
        self.code, self.details = code, details
        raise _AbortedError


def _request(env) -> pb2.CreateSessionRequest:
    return pb2.CreateSessionRequest(config=pb2.SessionConfig(project_path="/tmp", environment=env))


@pytest.mark.parametrize("env_of,status", [
    (lambda cli, other: {CLI_PATH_VAR: other, SANDBOX_ID_VAR: SBX_A}, grpc.StatusCode.PERMISSION_DENIED),
    (lambda cli, other: {CLI_PATH_VAR: cli, SANDBOX_ID_VAR: "nope"}, grpc.StatusCode.INVALID_ARGUMENT),
    (lambda cli, other: {SANDBOX_ID_VAR: SBX_A}, grpc.StatusCode.INVALID_ARGUMENT),
], ids=["not-allowed", "bad-id", "id-without-path"])
async def test_create_session_rpc_reports_a_refusal_as_the_callers_error(make_manager, cli, tmp_path, env_of, status):
    servicer = ClaudeAgentServicer(make_manager(cli))
    context = _Context()

    with pytest.raises(_AbortedError):
        await servicer.CreateSession(_request(env_of(cli, _executable(tmp_path / "other"))), context)

    assert context.code == status
    assert context.details


async def test_create_session_rpc_accepts_a_good_pair(make_manager, cli):
    servicer = ClaudeAgentServicer(make_manager(cli))
    session = await servicer.CreateSession(_request({CLI_PATH_VAR: cli, SANDBOX_ID_VAR: SBX_A}), _Context())
    assert session.session_id


def test_the_runner_process_environment_is_not_written(cli):
    before = dict(os.environ)
    session_cli_path(_config({CLI_PATH_VAR: cli, SANDBOX_ID_VAR: SBX_A}), frozenset({cli}))
    assert dict(os.environ) == before

