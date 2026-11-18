from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "provider"))

import client
import pytest

TEXT = ("assistantResponseEvent", {"content": "abcd"})
REASONING = ("reasoningContentEvent", {"text": "xyzxyz"})
TOOL = ("toolUseEvent", {"toolUseId": "t1", "name": "read_file", "input": '{"path":"x"}'})


def _percent(value):
    return {"contextUsagePercentage": value}


def _chunks(monkeypatch, events, model="claude-sonnet-5.5"):
    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_events", lambda _: iter(events))
    return list(instance._stream(model, {}))


def _usage(monkeypatch, events, model="claude-sonnet-5.5"):
    return getattr(_chunks(monkeypatch, events, model)[-1], "usage", None)


def test_stop_finish_carries_percentage_times_window_split_by_output(monkeypatch):
    """4 text + 6 reasoning chars are 4 completion tokens out of 10% of 1M."""
    chunks = _chunks(monkeypatch, [TEXT, REASONING, ("metadataEvent", {"stopReason": "END_TURN"}), ("contextUsageEvent", _percent(10.0))])
    assert chunks[-1].choices[0].finish_reason == "stop"
    usage = chunks[-1].usage
    assert (usage.total_tokens, usage.completion_tokens, usage.prompt_tokens) == (100_000, 4, 99_996)
    assert usage.prompt_tokens_details.cached_tokens == 0


def test_tool_calls_finish_carries_usage_from_event_after_the_stop_marker(monkeypatch):
    """The percentage arrives after metadataEvent TOOL_USE; name + arguments are 21 chars, 7 tokens."""
    chunks = _chunks(monkeypatch, [TOOL, ("metadataEvent", {"stopReason": "TOOL_USE"}), ("contextUsageEvent", _percent(10.0))])
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    assert chunks[-2].choices[0].delta.tool_calls[0].function.name == "read_file"
    assert (chunks[-1].usage.total_tokens, chunks[-1].usage.completion_tokens) == (100_000, 7)


def test_no_context_usage_event_leaves_zeros(monkeypatch):
    """Without a contextUsageEvent the stream chunk has no usage and `_complete` reports zeros."""
    events = [TEXT, ("metadataEvent", {"stopReason": "END_TURN"})]
    assert _usage(monkeypatch, events) is None
    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_events", lambda _: iter(events))
    usage = instance._complete("claude-sonnet-5.5", {}).usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (0, 0, 0)


def test_unknown_model_leaves_zeros(monkeypatch):
    """A model missing from the window table gets no invented numbers."""
    events = [TEXT, ("contextUsageEvent", _percent(10.0))]
    assert _usage(monkeypatch, events, model="not-a-kiro-model") is None
    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_events", lambda _: iter(events))
    assert instance._complete("not-a-kiro-model", {}).usage.total_tokens == 0


def test_percentage_is_percent_not_fraction(monkeypatch):
    """0.5 on a 200k window is 1000 tokens."""
    assert _usage(monkeypatch, [TEXT, ("contextUsageEvent", _percent(0.5))], model="claude-haiku-4.5").total_tokens == 1000


def test_last_context_usage_event_wins(monkeypatch):
    """A later contextUsageEvent replaces an earlier one."""
    events = [TEXT, ("contextUsageEvent", _percent(1.0)), ("contextUsageEvent", _percent(2.0))]
    assert _usage(monkeypatch, events).total_tokens == 20_000


def test_completion_tokens_never_exceed_total(monkeypatch):
    """Long output on a tiny total clamps completion to the total and leaves prompt at zero."""
    usage = _usage(monkeypatch, [("assistantResponseEvent", {"content": "x" * 300}), ("contextUsageEvent", _percent(0.0001))], model="claude-haiku-4.5")
    assert (usage.total_tokens, usage.completion_tokens, usage.prompt_tokens) == (0, 0, 0)
    usage = _usage(monkeypatch, [("assistantResponseEvent", {"content": "x" * 300}), ("contextUsageEvent", _percent(0.01))], model="claude-haiku-4.5")
    assert (usage.total_tokens, usage.completion_tokens, usage.prompt_tokens) == (20, 20, 0)


def test_complete_returns_stream_usage(monkeypatch):
    """The non-stream response carries the same usage as the final stream chunk."""
    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_events", lambda _: iter([TEXT, ("metadataEvent", {"stopReason": "END_TURN"}), ("contextUsageEvent", _percent(10.0))]))
    usage = instance._complete("claude-sonnet-5.5", {}).usage
    assert (usage.total_tokens, usage.completion_tokens, usage.prompt_tokens) == (100_000, 2, 99_998)
    assert usage.prompt_tokens_details.cached_tokens == 0


def test_every_model_in_the_table_has_a_positive_window():
    """Guards typos in the window table."""
    assert client._CONTEXT_WINDOWS
    for model, window in client._CONTEXT_WINDOWS.items():
        assert isinstance(window, int) and window > 0, model
        assert client._context_window(model) == window
    assert client._context_window("not-a-kiro-model") is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -5, -0.1, True, "1.5", None])
def test_unusable_percentage_gives_no_usage(value):
    """A non-finite, negative, boolean or non-numeric percentage yields no usage and never raises."""
    assert client._kiro_usage("claude-sonnet-5.5", value, 10) is None


def test_percentage_above_one_hundred_is_capped_at_the_window():
    """A percentage over 100 reports at most the model's window."""
    assert client._kiro_usage("claude-sonnet-4.5", 150.0, 0).total_tokens == 200000
