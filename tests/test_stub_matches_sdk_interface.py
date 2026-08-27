"""A test double may not invent methods the real SDK client does not have.

`send_prompt` called ``await client.send(prompt)``. `ClaudeSDKClient` has no
``send`` -- the method that writes a prompt into a streaming session is
``query`` -- so the FIRST turn of every Tier 2/Tier 3 session raised
``AttributeError`` and no prompt has ever reached Claude through that path.

The 25-test turn-boundary suite was green throughout, because `StubSDKClient`
defined ``send()``. A double is free to implement an interface nobody else has,
and once it does, the suite proves only that the double is self-consistent.
That is the failure this file exists to make impossible: it compares the double
and the call sites against the REAL class, so an API that does not exist fails
here instead of in a container.

Two directions, both needed:

* **call sites** -- what production code invokes on a client must exist.
* **the double** -- what the stub offers must exist too, or the next test
  written against it will encode another API that isn't real.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeSDKClient

from test_send_prompt_turn_boundary import StubSDKClient

SESSION_MANAGER = (
    Path(__file__).resolve().parents[1]
    / "src/claude_agent_grpc_server/sdk/session_manager.py"
)

#: Attributes on the double that are scaffolding, not SDK surface. Each one is
#: test-only state the harness inspects; none is ever called on a real client.
STUB_ONLY = {
    "script_turn",   # queues what the next query() should produce
    "emit",          # push one message without a prompt
    "close_stream",  # simulate the CLI transport going away
}


def _sdk_surface() -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(ClaudeSDKClient, callable)
        if not name.startswith("__")
    }


def _client_calls_in_source() -> set[str]:
    """Every ``client.<attr>(...)`` invoked in session_manager.

    AST rather than a regex: a regex would also match the word in a comment or
    a docstring, and this file has plenty of both discussing the very bug.
    """
    tree = ast.parse(SESSION_MANAGER.read_text(encoding="utf-8"))
    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in {"client", "summary_client"}
        ):
            calls.add(func.attr)
    return calls


def test_the_scan_finds_something():
    """Guard the guard: an AST walk that matches nothing passes vacuously."""
    calls = _client_calls_in_source()
    assert calls, "found no client.<method>() calls -- the scan is broken"
    assert "query" in calls, (
        "expected send_prompt to call client.query; if the call site moved, this "
        f"scan needs updating. Found: {sorted(calls)}"
    )


@pytest.mark.parametrize("attr", sorted(_client_calls_in_source()))
def test_every_method_we_call_exists_on_the_real_client(attr):
    assert attr in _sdk_surface(), (
        f"session_manager calls client.{attr}(), which ClaudeSDKClient does not "
        f"have. Real surface: {sorted(_sdk_surface())}"
    )


@pytest.mark.parametrize(
    "attr",
    sorted(
        name
        for name, _ in inspect.getmembers(StubSDKClient, callable)
        if not name.startswith("_")
    ),
)
def test_the_double_offers_no_method_the_real_client_lacks(attr):
    if attr in STUB_ONLY:
        pytest.skip(f"{attr} is declared test-only scaffolding")
    assert attr in _sdk_surface(), (
        f"StubSDKClient.{attr} does not exist on ClaudeSDKClient. Either the "
        f"real API changed, or the double is inventing one -- which is exactly "
        f"how client.send() survived. Add it to STUB_ONLY only if it is genuinely "
        f"harness scaffolding that no production code path calls."
    )


def test_stub_only_entries_are_real_attributes_of_the_stub():
    """Stop STUB_ONLY becoming a place to hide a typo'd exemption."""
    for name in STUB_ONLY:
        assert hasattr(StubSDKClient, name), (
            f"STUB_ONLY lists {name!r}, which StubSDKClient does not define -- a "
            f"stale exemption silently widens what this guard permits"
        )
