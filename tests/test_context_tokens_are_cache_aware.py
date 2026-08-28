"""A gRPC session must report the context it actually consumed.

`ResultMessage` carries `usage: dict` and has **no** `input_tokens` attribute.
The cost message was built with `getattr(message, 'input_tokens', None)`, which
therefore returned None on every turn; the proto conversion (`msg.input_tokens
or 0`) made that a confident 0, and the consuming client's
`if message.input_tokens:` never fired. Every gRPC-backed session reported a
context battery of 0% for its entire life, and every threshold keyed off it was
dead.

Measured on a live TASCIM Tier 3 container, 2026-08-28: a turn that cost $0.19
reported `input_tokens=0, output_tokens=0` on the wire, and the client's
context counter stayed at 0 across the whole session.

Reading `usage["input_tokens"]` alone is NOT the fix. Under prompt caching --
always on for SDK sessions -- that field is the uncached delta, routinely 1-2
tokens on a prompt of hundreds of thousands. It fails in the same silent,
safe-looking direction.
"""
from __future__ import annotations

import pytest

from claude_agent_grpc_server.sdk.session_manager import context_tokens_from_usage


class TestTheSumIsTheContext:
    def test_all_three_fields_are_added(self):
        assert context_tokens_from_usage({
            "input_tokens": 2,
            "cache_read_input_tokens": 313_618,
            "cache_creation_input_tokens": 146_539,
        }) == 460_159

    def test_reading_input_tokens_alone_would_be_wrong_by_orders_of_magnitude(self):
        """The regression guard, stated as the number it prevents."""
        usage = {
            "input_tokens": 2,
            "cache_read_input_tokens": 313_618,
            "cache_creation_input_tokens": 146_539,
        }
        assert context_tokens_from_usage(usage) > 200_000, (
            "a 460k-token prompt must not be reported as 2 tokens"
        )

    def test_output_tokens_are_not_context_input(self):
        """Output is generated, not carried into the next prompt as input here."""
        assert context_tokens_from_usage({
            "input_tokens": 100, "output_tokens": 9_999,
        }) == 100

    def test_an_uncached_turn_still_reports(self):
        assert context_tokens_from_usage({"input_tokens": 1_234}) == 1_234

    def test_cache_only_turn_reports_the_cache(self):
        assert context_tokens_from_usage({"cache_read_input_tokens": 5_000}) == 5_000


class TestItRefusesToInventANumber:
    """None means "nothing to say"; 0 would be a claim that the prompt was empty.

    That distinction is the whole bug: a transmitted 0 reads as a real reading
    and pins the battery, where an absent value lets the consumer keep its last.
    """

    @pytest.mark.parametrize("usage", [None, {}, "not a dict", 42, []],
                             ids=["none", "empty", "str", "int", "list"])
    def test_nothing_usable_yields_none(self, usage):
        assert context_tokens_from_usage(usage) is None

    @pytest.mark.parametrize("bad", [0, -1, None, "700", 12.5, float("nan")],
                             ids=["zero", "negative", "none", "str", "float", "nan"])
    def test_a_malformed_field_is_skipped_not_coerced(self, bad):
        """One bad field must not poison the fields that are fine."""
        assert context_tokens_from_usage({
            "input_tokens": bad, "cache_read_input_tokens": 700,
        }) == 700

    def test_all_fields_malformed_yields_none(self):
        assert context_tokens_from_usage({
            "input_tokens": None, "cache_read_input_tokens": "x",
            "cache_creation_input_tokens": -5,
        }) is None

    def test_unknown_keys_are_ignored(self):
        assert context_tokens_from_usage({
            "input_tokens": 10, "server_tool_use": {"web_search_requests": 3},
        }) == 10


class TestTheExcessEnd:
    """The numbers this reports are large and only get larger."""

    def test_a_million_token_window_is_not_truncated(self):
        assert context_tokens_from_usage({
            "input_tokens": 1, "cache_read_input_tokens": 999_999,
        }) == 1_000_000

    def test_it_stays_an_int_for_the_int64_wire_field(self):
        """The proto field is int64. A float here would fail at serialization."""
        val = context_tokens_from_usage({"input_tokens": 5, "cache_read_input_tokens": 5})
        assert isinstance(val, int) and not isinstance(val, bool)


class TestTheResultMessageShapeItWasWrittenAgainst:
    """Pin the SDK fact that caused the bug, so an SDK change surfaces here."""

    def test_result_message_has_no_input_tokens_attribute(self):
        from claude_agent_sdk import ResultMessage
        fields = set(getattr(ResultMessage, "__dataclass_fields__", {}))
        assert "usage" in fields, "usage is where the token counts live"
        assert "input_tokens" not in fields, (
            "if the SDK ever adds this attribute, re-check the extraction -- the "
            "old code read it and silently got None on every turn"
        )
