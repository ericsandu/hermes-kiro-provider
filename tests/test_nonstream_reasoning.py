from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "provider"))

import client


def _complete_from(monkeypatch, *deltas, finish="stop"):
    instance = client.KiroClient()
    chunks = [client._chunk("m", delta) for delta in deltas] + [client._chunk("m", {}, finish)]
    monkeypatch.setattr(instance, "_stream", lambda model, body: iter(chunks))
    return instance._complete("m", {}).choices[0].message


def test_complete_joins_reasoning_onto_both_message_fields(monkeypatch):
    """Reasoning deltas are joined into message.reasoning_content and message.reasoning."""
    message = _complete_from(monkeypatch, {"reasoning_content": "think "}, {"reasoning_content": "more"}, {"role": "assistant", "content": "answer"})
    assert message.reasoning_content == message.reasoning == "think more"


def test_complete_keeps_reasoning_out_of_content(monkeypatch):
    """Reasoning text never appears in message.content."""
    message = _complete_from(monkeypatch, {"reasoning_content": "secret thought"}, {"role": "assistant", "content": "answer"})
    assert message.content == "answer"


def test_complete_without_reasoning_leaves_none(monkeypatch):
    """A response with no reasoning deltas has None in both reasoning fields."""
    message = _complete_from(monkeypatch, {"role": "assistant", "content": "answer"})
    assert message.reasoning is None and message.reasoning_content is None


def test_complete_tool_call_response_keeps_tool_calls_with_reasoning(monkeypatch):
    """A tool-call turn with reasoning keeps its tool_calls and finish reason."""
    calls = client.KiroClient._tool_calls({"t1": {"name": "f", "arguments": "{}"}})
    message = _complete_from(monkeypatch, {"reasoning_content": "plan"}, {"role": "assistant", "tool_calls": calls}, finish="tool_calls")
    assert [call.function.name for call in message.tool_calls] == ["f"]
    assert message.reasoning_content == "plan" and message.content is None
