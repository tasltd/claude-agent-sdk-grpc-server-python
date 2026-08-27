"""Interrupt must end an in-flight streaming SendPrompt — TAS-422.

TAS-420 built interrupt-and-continue for Tier 2/3 on one unverified assumption:
that the gRPC `Interrupt` RPC terminates a **streaming** `send_prompt` mid-turn,
so the fused continuation prompt can be delivered. The ticket parked for two
months as "needs a live container".

It is not really a container question. `send_prompt` returns on `_TURN_END`, and
`_receive_loop` emits `_TURN_END` on `ResultMessage` and nowhere else -- so the
whole assumption reduces to *does the CLI produce a ResultMessage when
interrupted?*

Measured against the real CLI on 2026-08-27 (claude-agent-sdk 0.2.135): it does.
`interrupt()` returned at t+5.44s and a `ResultMessage(subtype=
'error_during_execution', is_error=True)` arrived at t+5.45s -- 1.45s after the
interrupt was requested. That is the fact these tests encode, so a future SDK
that stops emitting it fails here rather than hanging a Tier 2 session forever.

The interesting part is not the happy path but the SEAM: Tier2SessionService
polls `is_idle()`, which the server flips to "idle" the instant interrupt() is
called -- roughly 1.4s BEFORE the ResultMessage lands. So the next prompt is
routinely sent while the interrupted turn's `_TURN_END` is still in flight, and
`_drain_stale` cannot help because it ran before that message existed.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

from claude_agent_grpc_server.sdk.session_manager import (
    SessionConfig,
    SessionInfo,
    SessionManager,
)

from tests.test_send_prompt_turn_boundary import (  # reuse, do not re-invent
    StubSDKClient,
    _assistant,
    _install_session,
    manager,  # noqa: F401 -- pytest fixture
)


def _interrupted_result() -> ResultMessage:
    """The ResultMessage the real CLI emits for an interrupted turn.

    subtype and is_error are the measured values, not invented ones. They matter:
    `_convert_message` does not read is_error, which is why an interrupt reports
    status="completed" rather than surfacing as a session error.
    """
    return ResultMessage(
        subtype="error_during_execution",
        duration_ms=1450,
        duration_api_ms=1400,
        is_error=True,
        num_turns=1,
        session_id="sdk-session",
        total_cost_usd=0.002,
    )


async def _drain(agen, timeout: float = 2.0):
    out = []
    async def run():
        async for m in agen:
            out.append(m)
    await asyncio.wait_for(run(), timeout=timeout)
    return out


@pytest.mark.asyncio
async def test_interrupt_result_ends_the_in_flight_rpc(manager):  # noqa: F811
    """The RPC returns once the interrupted turn's ResultMessage arrives."""
    client = StubSDKClient()
    _install_session(manager, client)
    client.script_turn([_assistant("counting 1")])  # no result: turn stays open

    agen = manager.send_prompt("s1", "count to 400")
    task = asyncio.create_task(_drain(agen, timeout=3.0))
    await asyncio.sleep(0.05)
    assert not task.done(), "RPC ended before the turn did"

    assert await manager.interrupt("s1") is True
    assert client.interrupted == 1

    # The CLI answers the interrupt ~1.4s later with a ResultMessage.
    client.emit(_interrupted_result())
    messages = await asyncio.wait_for(task, timeout=3.0)

    assert any(m.type == "text" and "counting 1" in (m.content or "") for m in messages)
    assert any(m.type == "status" and m.status == "completed" for m in messages)


@pytest.mark.asyncio
async def test_an_interrupt_is_not_reported_as_a_session_error(manager):  # noqa: F811
    """is_error=True on the ResultMessage must NOT surface as an error message.

    An interrupt is a user action, not a fault. `_convert_message` ignores
    is_error, so the turn closes as "completed" -- which is right here, and is
    pinned so nobody "fixes" it into an error banner on every Ctrl+Enter.

    The cost of that choice, recorded deliberately: a GENUINE
    error_during_execution (a real CLI failure, not an interrupt) also reports
    completed. Distinguishing them needs a signal this layer does not have --
    only the caller knows whether it asked for the interrupt.
    """
    client = StubSDKClient()
    info = _install_session(manager, client)
    client.script_turn([_assistant("working")])

    task = asyncio.create_task(_drain(manager.send_prompt("s1", "go"), timeout=3.0))
    await asyncio.sleep(0.05)
    await manager.interrupt("s1")
    client.emit(_interrupted_result())
    messages = await asyncio.wait_for(task, timeout=3.0)

    assert not any(m.is_error for m in messages), \
        "an interrupted turn must not emit an error message"
    assert info.status == "idle"


@pytest.mark.asyncio
async def test_interrupt_marks_idle_before_the_result_arrives(manager):  # noqa: F811
    """Pins the race Tier2SessionService actually runs into.

    interrupt() sets status="idle" immediately, so `is_idle()` is true while the
    previous turn's ResultMessage is still ~1.4s away. Tier 2 polls exactly this
    and will therefore send the continuation prompt early -- by design, and the
    next test proves that is safe.
    """
    client = StubSDKClient()
    info = _install_session(manager, client)
    client.script_turn([_assistant("working")])

    task = asyncio.create_task(_drain(manager.send_prompt("s1", "go"), timeout=3.0))
    await asyncio.sleep(0.05)
    assert info.status == "running"

    await manager.interrupt("s1")
    assert info.status == "idle", "is_idle() must be true before the ResultMessage"
    assert not task.done(), "the RPC has NOT ended yet -- only the status moved"

    client.emit(_interrupted_result())
    await asyncio.wait_for(task, timeout=3.0)


@pytest.mark.asyncio
async def test_the_continuation_turn_does_not_eat_the_interrupts_turn_end(manager):  # noqa: F811
    """The continuation prompt sent early must still get its OWN output.

    Real ordering, and the one `_drain_stale` cannot cover:

      1. turn 1 streaming, RPC A parked on the queue
      2. interrupt() -> status idle
      3. Tier 2 sees idle and sends the fused continuation -> RPC B starts and
         drains a queue that is still EMPTY
      4. only now does the interrupt's ResultMessage arrive
      5. turn 2's real output follows

    If B consumed that stale `_TURN_END` it would return empty and the
    continuation would look like a silent no-op. It does not, because
    asyncio.Queue wakes getters in FIFO order and A has been waiting longer.
    That ordering is load-bearing and undocumented at the call site, which is
    the entire reason this test exists.
    """
    client = StubSDKClient()
    _install_session(manager, client)
    client.script_turn([_assistant("counting 1")])

    task_a = asyncio.create_task(_drain(manager.send_prompt("s1", "count to 400"), timeout=3.0))
    await asyncio.sleep(0.05)

    await manager.interrupt("s1")

    # Continuation goes out BEFORE the interrupt's result lands.
    task_b = asyncio.create_task(
        _drain(manager.send_prompt("s1", "[continuing] now summarise instead"), timeout=3.0)
    )
    await asyncio.sleep(0.05)
    assert not task_b.done(), "the continuation RPC returned before producing anything"

    client.emit(_interrupted_result())          # step 4
    await asyncio.sleep(0.05)
    a_messages = await asyncio.wait_for(task_a, timeout=3.0)

    client.emit(_assistant("here is the summary"))   # step 5
    client.emit(ResultMessage(
        subtype="success", duration_ms=10, duration_api_ms=8, is_error=False,
        num_turns=2, session_id="sdk-session", total_cost_usd=0.01,
    ))
    b_messages = await asyncio.wait_for(task_b, timeout=3.0)

    assert any("counting 1" in (m.content or "") for m in a_messages)
    assert any("here is the summary" in (m.content or "") for m in b_messages), \
        "the continuation turn returned without its own output -- it ate the stale _TURN_END"
    assert not any("here is the summary" in (m.content or "") for m in a_messages)
    assert client.sent[-1].startswith("[continuing]")


@pytest.mark.asyncio
async def test_interrupt_on_an_unknown_session_raises(manager):  # noqa: F811
    with pytest.raises(ValueError, match="Session not found"):
        await manager.interrupt("no-such-session")
