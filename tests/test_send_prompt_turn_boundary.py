"""A SendPrompt RPC must end when the turn ends — TAS-624.

`send_prompt` used to iterate `client.receive_messages()` itself with no break
on ResultMessage. `receive_messages()` ends only when the CLI transport closes,
and a `ClaudeSDKClient` keeps its CLI alive across turns by design, so the
generator parked forever. Four defects followed, and every class below pins one
of them so it cannot come back quietly:

1. the RPC never completed, so `status = "idle"` and `save_session_outputs()`
   after the loop were unreachable;
2. the session therefore reported RUNNING forever, and `Tier2Session.is_idle()`
   reads that status;
3. turn 2 onward opened a SECOND consumer of the same anyio memory object
   stream — which delivers each message to exactly one receiver — so roughly
   half of each later turn went to the abandoned iterator;
4. whenever the generator was suspended at a `yield`, nothing drained the SDK's
   100-slot buffer; once full the SDK's reader task parks and takes
   `interrupt()`, `set_model()` and every hook callback with it, silently.

The fix is a persistent consumer per session feeding a queue the RPC drains for
one turn. These tests exercise the real `SessionManager` against a stub SDK
client, so they fail if the consumer is ever moved back inside the RPC.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

from claude_agent_grpc_server.sdk.session_manager import (
    _MAX_UNDRAINED_MESSAGES,
    _TURN_END,
    SessionConfig,
    SessionInfo,
    SessionManager,
    StreamMessage,
)

# --------------------------------------------------------------------------
# The messages are the REAL SDK dataclasses. `_convert_message` dispatches on
# isinstance, so hand-rolled look-alikes are silently ignored — a stub message
# would make every test below hang rather than fail, which is exactly the
# failure mode this file exists to catch.
# --------------------------------------------------------------------------

def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], model="claude-opus-5")


def _result() -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id="sdk-session",
        total_cost_usd=0.01,
    )


class StubSDKClient:
    """Feeds messages on demand and NEVER closes its stream by itself.

    That last property is the point: a stub whose iterator ends after the queued
    messages would make the old buggy code pass, because falling off the end
    looks exactly like a turn boundary.
    """

    def __init__(self) -> None:
        self._outbox: asyncio.Queue = asyncio.Queue()
        self._script: list[list[Any]] = []
        self.sent: list[str] = []
        self.interrupted = 0
        self.disconnected = False
        self.closed = False

    def script_turn(self, messages: list[Any]) -> None:
        """Queue what the NEXT send() should produce.

        The real CLI emits a turn's output only after it receives the prompt.
        Emitting beforehand would be output belonging to no turn, which
        send_prompt deliberately discards -- so a test that pre-emitted would
        be measuring the wrong thing.
        """
        self._script.append(list(messages))

    async def connect(self) -> None:  # pragma: no cover - trivial
        return None

    async def disconnect(self) -> None:
        self.disconnected = True

    async def send(self, prompt: str) -> None:
        self.sent.append(prompt)
        if self._script:
            for message in self._script.pop(0):
                self._outbox.put_nowait(message)

    async def interrupt(self) -> None:
        self.interrupted += 1

    def emit(self, message: Any) -> None:
        self._outbox.put_nowait(message)

    def close_stream(self) -> None:
        """Simulate the CLI transport going away."""
        self._outbox.put_nowait(_STREAM_CLOSED)

    async def receive_messages(self):
        while True:
            item = await self._outbox.get()
            if item is _STREAM_CLOSED:
                self.closed = True
                return
            yield item


_STREAM_CLOSED = object()


def _turn(text: str = "hello") -> list[Any]:
    """One assistant message followed by the ResultMessage that ends the turn."""
    return [_assistant(text), _result()]


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """A real SessionManager with its on-disk state redirected into tmp_path."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return SessionManager()


def _install_session(manager: SessionManager, client: StubSDKClient,
                     session_id: str = "s1") -> SessionInfo:
    config = SessionConfig(project_path="/tmp", model="claude-opus-5")
    info = SessionInfo(
        id=session_id,
        name=session_id,
        status="idle",
        project_path="/tmp",
        model="claude-opus-5",
        created_at=datetime.now(timezone.utc),
    )
    manager._sessions[session_id] = (info, config, client)
    return info


async def _collect(agen, timeout: float = 2.0) -> list[StreamMessage]:
    """Drain an async generator to completion, failing loudly on a hang.

    A plain `async for` here would HANG rather than fail under the old code,
    which is a far worse test outcome than a red assertion.
    """
    out: list[StreamMessage] = []

    async def run():
        async for msg in agen:
            out.append(msg)

    await asyncio.wait_for(run(), timeout=timeout)
    return out


# --------------------------------------------------------------------------


class TestATurnEnds:
    async def test_send_prompt_returns_at_the_result_message(self, manager):
        client = StubSDKClient()
        _install_session(manager, client)
        client.script_turn(_turn())

        msgs = await _collect(manager.send_prompt("s1", "hi"))

        assert any(m.type == "text" for m in msgs)
        assert client.sent == ["hi"]

    async def test_it_does_not_wait_for_the_transport_to_close(self, manager):
        """The regression itself. The stub's stream is still open afterwards."""
        client = StubSDKClient()
        _install_session(manager, client)
        client.script_turn(_turn())

        await _collect(manager.send_prompt("s1", "hi"))

        assert client.closed is False, (
            "the RPC only ended because the stream closed — that is the old bug"
        )

    async def test_the_session_returns_to_idle(self, manager):
        """Tier2Session.is_idle() reads this. It used to stay RUNNING forever."""
        client = StubSDKClient()
        info = _install_session(manager, client)
        client.script_turn(_turn())

        await _collect(manager.send_prompt("s1", "hi"))
        await asyncio.sleep(0)

        assert info.status == "idle"

    async def test_outputs_are_saved_at_the_turn_boundary(self, manager):
        """Unreachable before the fix: it sat after the never-ending loop."""
        client = StubSDKClient()
        _install_session(manager, client)
        client.script_turn(_turn())

        await _collect(manager.send_prompt("s1", "hi"))

        assert manager._session_outputs.get("s1"), "nothing persisted"


class TestSuccessiveTurnsDoNotSplitTheirOutput:
    async def test_turn_two_gets_all_of_its_own_output(self, manager):
        """Defect 3: two live consumers of one stream each got about half."""
        client = StubSDKClient()
        _install_session(manager, client)

        client.script_turn(_turn("first"))
        first = await _collect(manager.send_prompt("s1", "one"))

        client.script_turn(_turn("second"))
        second = await _collect(manager.send_prompt("s1", "two"))

        assert [m.content for m in first if m.type == "text"] == ["first"]
        assert [m.content for m in second if m.type == "text"] == ["second"]

    async def test_only_one_consumer_task_ever_exists(self, manager):
        client = StubSDKClient()
        _install_session(manager, client)

        for _ in range(3):
            client.script_turn(_turn())
            await _collect(manager.send_prompt("s1", "go"))

        assert len(manager._receive_tasks) == 1

    async def test_five_turns_all_terminate(self, manager):
        """Multiplicity: the old code hung on turn 1, so N was never exercised."""
        client = StubSDKClient()
        _install_session(manager, client)

        for i in range(5):
            client.script_turn(_turn(f"turn-{i}"))
            msgs = await _collect(manager.send_prompt("s1", f"p{i}"))
            assert [m.content for m in msgs if m.type == "text"] == [f"turn-{i}"]


class TestTheSdkStreamAlwaysHasAConsumer:
    """Defect 4 — the one with no error surface."""

    async def test_the_consumer_outlives_the_rpc(self, manager):
        client = StubSDKClient()
        _install_session(manager, client)
        client.script_turn(_turn())
        await _collect(manager.send_prompt("s1", "hi"))

        task = manager._receive_tasks["s1"]
        assert not task.done(), (
            "the consumer stopped with the RPC — between turns nothing drains "
            "the SDK buffer, which is what kills interrupt() and the hooks"
        )

    async def test_messages_arriving_between_turns_are_still_consumed(self, manager):
        """The gap the old design could not cover at all."""
        client = StubSDKClient()
        _install_session(manager, client)
        client.script_turn(_turn())
        await _collect(manager.send_prompt("s1", "hi"))

        client.emit(_assistant("late"))
        await asyncio.sleep(0.05)

        persisted = [o.content for o in manager._session_outputs["s1"]]
        assert "late" in persisted

    async def test_a_conversion_failure_does_not_stop_the_loop(self, manager,
                                                              monkeypatch):
        """Our own bug must not strand the SDK's buffer."""
        client = StubSDKClient()
        _install_session(manager, client)

        real = manager._convert_message
        calls = {"n": 0}

        def flaky(message, session_info):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return real(message, session_info)

        monkeypatch.setattr(manager, "_convert_message", flaky)

        client.script_turn([_assistant("poison"), *_turn("survivor")])

        msgs = await _collect(manager.send_prompt("s1", "hi"))

        assert [m.content for m in msgs if m.type == "text"] == ["survivor"]
        assert not manager._receive_tasks["s1"].done()

    async def test_a_persistence_failure_does_not_stop_the_loop(self, manager,
                                                               monkeypatch):
        client = StubSDKClient()
        _install_session(manager, client)
        monkeypatch.setattr(
            manager, "_persist_output",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))

        client.script_turn(_turn("still-delivered"))
        msgs = await _collect(manager.send_prompt("s1", "hi"))

        assert [m.content for m in msgs if m.type == "text"] == ["still-delivered"]

    async def test_a_dead_consumer_is_restarted_rather_than_fatal(self, manager):
        """A session with no consumer is the state that strands the buffer."""
        client = StubSDKClient()
        _install_session(manager, client)
        client.script_turn(_turn())
        await _collect(manager.send_prompt("s1", "hi"))

        manager._receive_tasks["s1"].cancel()
        await asyncio.sleep(0.01)

        client.script_turn(_turn("after-restart"))
        msgs = await _collect(manager.send_prompt("s1", "again"))

        assert [m.content for m in msgs if m.type == "text"] == ["after-restart"]


class TestTheRpcIsAlwaysReleased:
    """Trading a never-ending turn for a differently-never-ending one is no fix."""

    async def test_a_closed_transport_ends_the_rpc_with_an_error(self, manager):
        client = StubSDKClient()
        _install_session(manager, client)

        async def collect():
            return await _collect(manager.send_prompt("s1", "hi"))

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.02)
        client.close_stream()

        msgs = await asyncio.wait_for(task, timeout=2.0)
        assert any(m.is_error for m in msgs)

    async def test_deleting_a_session_releases_a_waiting_rpc(self, manager):
        client = StubSDKClient()
        _install_session(manager, client)

        async def collect():
            return await _collect(manager.send_prompt("s1", "hi"), timeout=3.0)

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.02)
        await manager.delete_session("s1")

        await asyncio.wait_for(task, timeout=2.0)

    async def test_delete_session_cancels_the_consumer(self, manager):
        client = StubSDKClient()
        _install_session(manager, client)
        client.script_turn(_turn())
        await _collect(manager.send_prompt("s1", "hi"))
        task = manager._receive_tasks["s1"]

        await manager.delete_session("s1")
        await asyncio.sleep(0.01)

        assert task.cancelled() or task.done()
        assert "s1" not in manager._receive_tasks
        assert "s1" not in manager._turn_queues


class TestStaleOutputIsNotReplayed:
    async def test_a_new_prompt_discards_a_previous_turns_leftovers(self, manager):
        """Otherwise one prompt's output is billed to the next."""
        client = StubSDKClient()
        _install_session(manager, client)
        queue = manager._ensure_receive_loop("s1", client)
        queue.put_nowait(StreamMessage(type="text", content="orphan"))

        client.script_turn(_turn("fresh"))
        msgs = await _collect(manager.send_prompt("s1", "hi"))

        assert [m.content for m in msgs if m.type == "text"] == ["fresh"]

    async def test_drain_stale_reports_what_it_dropped(self):
        q: asyncio.Queue = asyncio.Queue()
        for _ in range(4):
            q.put_nowait(StreamMessage(type="text", content="x"))
        assert SessionManager._drain_stale(q) == 4
        assert q.empty()

    async def test_drain_stale_on_an_empty_queue_is_zero(self):
        assert SessionManager._drain_stale(asyncio.Queue()) == 0


class TestExcess:
    """The overloaded end — where a bounded queue would re-introduce the bug."""

    async def test_the_queue_never_blocks_the_consumer(self, manager):
        """If `_offer` ever awaits, the SDK's reader parks and the control
        plane dies silently. Enqueue far past the cap and prove otherwise."""
        client = StubSDKClient()
        _install_session(manager, client)
        queue = manager._ensure_receive_loop("s1", client)

        for i in range(_MAX_UNDRAINED_MESSAGES * 2):
            manager._offer("s1", queue, StreamMessage(type="text", content=str(i)))

        assert queue.qsize() <= _MAX_UNDRAINED_MESSAGES + 1
        assert manager._dropped_messages["s1"] > 0

    async def test_a_turn_boundary_is_never_dropped(self, manager):
        """Dropping one would hang the next RPC forever — strictly worse."""
        client = StubSDKClient()
        _install_session(manager, client)
        queue = manager._ensure_receive_loop("s1", client)

        for i in range(_MAX_UNDRAINED_MESSAGES + 50):
            manager._offer("s1", queue, StreamMessage(type="text", content=str(i)))
        manager._offer("s1", queue, _TURN_END)

        drained = []
        while not queue.empty():
            drained.append(queue.get_nowait())
        assert _TURN_END in drained

    async def test_a_long_turn_delivers_every_message(self, manager):
        """Ten times the design count, with a reader present: nothing is lost."""
        client = StubSDKClient()
        _install_session(manager, client)
        client.script_turn([_assistant(f"m{i}") for i in range(500)] + [_result()])

        msgs = await _collect(manager.send_prompt("s1", "hi"), timeout=10.0)

        texts = [m.content for m in msgs if m.type == "text"]
        assert len(texts) == 500
        assert texts[0] == "m0" and texts[-1] == "m499"

    async def test_the_drop_counter_resets_each_turn(self, manager):
        client = StubSDKClient()
        _install_session(manager, client)
        queue = manager._ensure_receive_loop("s1", client)
        for i in range(_MAX_UNDRAINED_MESSAGES + 5):
            manager._offer("s1", queue, StreamMessage(type="text", content=str(i)))
        assert manager._dropped_messages["s1"] > 0

        client.script_turn(_turn())
        await _collect(manager.send_prompt("s1", "hi"), timeout=5.0)

        assert manager._dropped_messages.get("s1", 0) == 0


class TestTheseTestsWouldHaveCaughtIt:
    """A regression suite that cannot fail is worse than none."""

    async def test_the_stub_stream_really_does_not_end_at_a_result(self):
        """If it did, the old code would pass and this file would prove nothing."""
        client = StubSDKClient()
        client.emit(_result())

        seen = []

        async def read():
            async for m in client.receive_messages():
                seen.append(m)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(read(), timeout=0.2)
        assert len(seen) == 1

    async def test_the_collect_helper_fails_instead_of_hanging(self):
        async def forever():
            await asyncio.sleep(10)
            yield StreamMessage(type="text", content="never")

        with pytest.raises(asyncio.TimeoutError):
            await _collect(forever(), timeout=0.2)

    async def test_an_unknown_session_is_rejected(self, manager):
        with pytest.raises(ValueError):
            await _collect(manager.send_prompt("nope", "hi"))
