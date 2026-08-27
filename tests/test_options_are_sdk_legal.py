"""The options SendPrompt builds must survive the SDK's own command builder.

Found by running the server in a container against a real Claude CLI, not by
any unit test: every prompt died in milliseconds with

    TypeError: 'NoneType' object is not iterable

`send_prompt` passed ``allowed_tools=None`` whenever the session had no
allow-list -- the common case for a Tier 2/Tier 3 session. That was always out
of contract (``ClaudeAgentOptions.allowed_tools`` is declared ``list[str]`` with
``default_factory=list``) but stayed harmless for as long as every consumer of
the field sat behind a truthiness check.

The Skills feature added the first unguarded one:
``subprocess_cli._apply_skills_defaults`` does ``list(self._options.allowed_tools)``
and ``_build_command`` calls it on EVERY connect. So a value that had merely
been wrong became fatal on an SDK upgrade, before the CLI was even spawned.

Asserting ``options.allowed_tools == []`` would pin today's symptom. These tests
instead push the options the REAL code path builds through the SDK's own
builder, so the next field handed a None -- or the next unguarded consumer the
SDK adds -- fails here rather than in a container.
"""
from __future__ import annotations

import pytest
from claude_agent_sdk.types import ClaudeAgentOptions

from claude_agent_grpc_server.sdk import session_manager as sm
from claude_agent_grpc_server.sdk.session_manager import SessionConfig

from test_send_prompt_turn_boundary import (  # noqa: E402  (shared harness)
    StubSDKClient,
    _collect,
    _install_session,
    _turn,
    manager,  # noqa: F401  (fixture re-export)
)


def _build_command(options: ClaudeAgentOptions) -> list[str]:
    """Run the SDK's real argv builder over these options.

    Reached by private path on purpose: this test exists precisely to notice
    when that internal changes shape under us.

    ``_cli_path`` is stubbed rather than discovered: the builder refuses to run
    until connect() has resolved a CLI, and what is under test here is argv
    assembly, not CLI discovery. A real path would make the test depend on
    whether a Claude binary happens to be installed on the runner.
    """
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    transport = SubprocessCLITransport(prompt="probe", options=options)
    transport._cli_path = "/nonexistent/claude"
    return transport._build_command()


@pytest.fixture
def recorded_options(monkeypatch):
    """Capture what send_prompt hands ClaudeAgentOptions, building the real thing.

    Reading the options off the production code path rather than re-deriving
    them in the test: a helper that rebuilt them here would keep passing after
    the real call site drifted, which is the failure this whole file is about.
    """
    seen: list[ClaudeAgentOptions] = []
    real = sm.ClaudeAgentOptions

    def recorder(**kwargs):
        built = real(**kwargs)
        seen.append(built)
        return built

    monkeypatch.setattr(sm, "ClaudeAgentOptions", recorder)
    return seen


def test_the_sdk_builder_is_actually_reachable():
    """Guard the guard: if this import breaks, everything below stops testing.

    Without it a renamed SDK internal would turn each assertion below into a
    collection error, and a suite that errors is not a suite that checked.
    """
    assert _build_command(ClaudeAgentOptions(model="sonnet")), (
        "the SDK command builder returned nothing"
    )


def test_none_allowed_tools_is_the_regression_and_still_raises():
    """Pin the mechanism, so the comment at the call site cannot rot silently.

    If a future SDK starts tolerating None this fails, which tells us the hazard
    is gone. That is worth knowing and is NOT the same as the fix being
    unnecessary -- the declared type still says list.
    """
    with pytest.raises(TypeError, match="NoneType"):
        _build_command(
            ClaudeAgentOptions(model="sonnet", allowed_tools=None)  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "allowed, disallowed",
    [
        ([], []),                       # the case that broke: no allow-list
        ([], ["Bash"]),                 # deny-only, how T2/T3 is actually used
        (["Read"], []),                 # allow-list present
        (["Read"], ["Bash", "Write"]),  # both
    ],
    ids=["neither", "deny-only", "allow-only", "both"],
)
async def test_every_tool_list_shape_builds_a_command(
    manager, recorded_options, allowed, disallowed  # noqa: F811
):
    client = StubSDKClient()
    info = _install_session(manager, client)
    manager._sessions["s1"] = (
        info,
        SessionConfig(
            project_path="/tmp",
            model="claude-opus-5",
            allowed_tools=list(allowed),
            disallowed_tools=list(disallowed),
        ),
        client,
    )
    client.script_turn(_turn())

    msgs = await _collect(manager.send_prompt("s1", "hi"))

    assert not [m for m in msgs if m.is_error], [m.content for m in msgs if m.is_error]
    assert recorded_options, "send_prompt never built any options"
    options = recorded_options[-1]

    # The declared types, honoured, whatever the config held.
    assert isinstance(options.allowed_tools, list)
    assert isinstance(options.disallowed_tools, list)

    # And the SDK agrees -- the assertion that actually caught this.
    _build_command(options)


async def test_a_prompt_with_no_allow_list_reaches_the_client(manager):  # noqa: F811
    """The end-to-end shape a Tier 2/Tier 3 session uses: deny-list, no allow-list."""
    client = StubSDKClient()
    info = _install_session(manager, client)
    manager._sessions["s1"] = (
        info,
        SessionConfig(
            project_path="/tmp",
            model="claude-opus-5",
            allowed_tools=[],
            disallowed_tools=["Bash", "Write", "Edit"],
        ),
        client,
    )
    client.script_turn(_turn("ALPHA"))

    msgs = await _collect(manager.send_prompt("s1", "hi"))

    assert client.sent == ["hi"], "the prompt never reached the SDK client"
    assert not [m for m in msgs if m.is_error], [m.content for m in msgs if m.is_error]
    assert any("ALPHA" in (m.content or "") for m in msgs)
