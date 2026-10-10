"""A session's own environment reaches its CLI (TAS-1044).

The host sends ``SessionConfig.environment`` with every CreateSession. The
server stored it and persisted it, but never applied it: ``send_prompt`` built
``ClaudeAgentOptions`` with no ``env=``. So a host could not point one session's
CLI at its account router (``ANTHROPIC_BASE_URL``) without writing a credential
into this process's environment, which every other session then inherits and
which the agent can read back.

These tests read the options off the real ``send_prompt`` code path, the same
way ``test_options_are_sdk_legal.py`` does.
"""
from __future__ import annotations

import pytest
from claude_agent_sdk.types import ClaudeAgentOptions
from test_options_are_sdk_legal import _build_command
from test_send_prompt_turn_boundary import (  # noqa: E402  (shared harness)
    StubSDKClient,
    _collect,
    _install_session,
    _turn,
    manager,  # noqa: F401  (fixture re-export)
)

from claude_agent_grpc_server.sdk import session_manager as sm
from claude_agent_grpc_server.sdk.session_manager import SessionConfig, session_cli_env

ROUTED = {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:18765/r/k-one",
    "CLAUDE_CODE_OAUTH_TOKEN": "tascim-route-k-one",
}


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


async def _prompt(manager, session_id: str, environment) -> ClaudeAgentOptions:  # noqa: F811
    client = StubSDKClient()
    info = _install_session(manager, client, session_id=session_id)
    config = SessionConfig(project_path="/tmp", model="claude-opus-5")
    if environment is not None:
        config.environment = environment
    manager._sessions[session_id] = (info, config, client)
    client.script_turn(_turn())
    msgs = await _collect(manager.send_prompt(session_id, "hi"))
    assert not [m for m in msgs if m.is_error], [m.content for m in msgs if m.is_error]
    return client


async def test_the_session_environment_is_on_the_options(manager, recorded_options):  # noqa: F811
    await _prompt(manager, "s1", dict(ROUTED))

    assert recorded_options, "send_prompt never built any options"
    assert recorded_options[-1].env == ROUTED
    _build_command(recorded_options[-1])


@pytest.mark.parametrize("environment", [None, {}], ids=["missing", "empty"])
async def test_no_environment_is_an_empty_dict(manager, recorded_options, environment):  # noqa: F811
    await _prompt(manager, "s1", environment)

    assert recorded_options[-1].env == {}


def test_a_none_environment_is_an_empty_dict():
    config = SessionConfig(project_path="/tmp")
    config.environment = None  # type: ignore[assignment]

    assert session_cli_env(config) == {}


def test_values_are_strings():
    config = SessionConfig(project_path="/tmp", environment={"A": 1, "B": True})  # type: ignore[dict-item]

    assert session_cli_env(config) == {"A": "1", "B": "True"}


async def test_a_large_environment_arrives_whole(manager, recorded_options):  # noqa: F811
    big = {f"VAR_{i:04d}": "x" * 2000 for i in range(500)}
    big.update(ROUTED)

    await _prompt(manager, "s1", big)

    assert recorded_options[-1].env == big


async def test_each_session_gets_only_its_own_environment(manager, recorded_options):  # noqa: F811
    envs = {
        f"s{i}": {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:18765/r/key-{i}",
            "CLAUDE_CODE_OAUTH_TOKEN": f"tascim-route-key-{i}",
        }
        for i in range(25)
    }
    for session_id, env in envs.items():
        await _prompt(manager, session_id, dict(env))

    assert [o.env for o in recorded_options] == list(envs.values())


async def test_the_process_environment_is_not_written(manager, recorded_options, monkeypatch):  # noqa: F811
    """The base URL belongs to one session's CLI, not to this whole process."""
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    import os

    await _prompt(manager, "s1", dict(ROUTED))

    assert "ANTHROPIC_BASE_URL" not in os.environ


async def test_the_summary_call_uses_the_session_environment(manager, monkeypatch):  # noqa: F811
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
    manager._sessions["s1"] = (info, SessionConfig(project_path="/tmp", environment=dict(ROUTED)), client)
    manager._session_outputs["s1"] = [
        sm.SessionOutput(timestamp="2026-10-10T00:00:00Z", message_type="text", content="did a thing " * 20)
    ]

    await manager.generate_summary("s1")

    assert built and built[-1].env == ROUTED
