from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "provider"))

import client
from credentials import KiroAuthError


def _stream_with(monkeypatch, events):
    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_events", lambda _: iter(events))
    return instance


def test_reasoning_events_stream_before_text(monkeypatch):
    """Reasoning arrives as reasoning_content chunks ahead of the answer text."""
    instance = _stream_with(monkeypatch, [
        ("reasoningContentEvent", {"text": "think"}),
        ("reasoningContentEvent", {"text": "more"}),
        ("assistantResponseEvent", {"content": "answer"}),
    ])
    chunks = list(instance._stream_impl("claude-sonnet-4.5", {}))
    deltas = [chunk.choices[0].delta for chunk in chunks]
    assert [getattr(d, "reasoning_content", None) for d in deltas[:2]] == ["think", "more"]
    assert deltas[2].content == "answer" and deltas[2].role == "assistant"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_reasoning_text_never_enters_content(monkeypatch):
    """Reasoning chunks carry no content field."""
    instance = _stream_with(monkeypatch, [
        ("reasoningContentEvent", {"text": "secret"}),
        ("assistantResponseEvent", {"content": "ok"}),
    ])
    deltas = [chunk.choices[0].delta for chunk in instance._stream_impl("claude-sonnet-4.5", {})]
    assert not hasattr(deltas[0], "content")
    assert "secret" not in "".join(getattr(d, "content", "") or "" for d in deltas)


def test_reasoning_does_not_claim_the_assistant_role(monkeypatch):
    """The first text chunk after reasoning still carries the assistant role."""
    instance = _stream_with(monkeypatch, [
        ("reasoningContentEvent", {"text": "hm"}),
        ("assistantResponseEvent", {"content": "a"}),
        ("assistantResponseEvent", {"content": "b"}),
    ])
    deltas = [chunk.choices[0].delta for chunk in instance._stream_impl("claude-sonnet-4.5", {})]
    assert [getattr(d, "role", "absent") for d in deltas[:3]] == ["absent", "assistant", None]


def test_reasoning_then_tool_use_keeps_tool_call_and_finish(monkeypatch):
    """A stream of reasoning followed by a tool use ends with the tool call and tool_calls finish."""
    instance = _stream_with(monkeypatch, [
        ("reasoningContentEvent", {"text": "plan"}),
        ("toolUseEvent", {"toolUseId": "t1", "name": "ls", "input": "{}"}),
        ("metadataEvent", {"stopReason": "TOOL_USE"}),
    ])
    chunks = list(instance._stream_impl("claude-sonnet-4.5", {}))
    assert chunks[1].choices[0].delta.tool_calls[0].function.name == "ls"
    assert chunks[1].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_reasoning_only_stream_is_still_an_empty_stream_error(monkeypatch):
    """Reasoning without an answer is not a completed reply, so the empty-stream error stands."""
    instance = _stream_with(monkeypatch, [("reasoningContentEvent", {"text": "hmm"})])
    with pytest.raises(KiroAuthError, match="without an event"):
        list(instance._stream_impl("claude-sonnet-4.5", {}))


def _frame(payload):
    event_type = b"assistantResponseEvent"
    headers = b"\x0b:event-type\x07" + struct.pack(">H", len(event_type)) + event_type
    prelude = struct.pack(">II", 16 + len(headers) + len(payload), len(headers))
    prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    frame = prelude + headers + payload
    return frame + struct.pack(">I", zlib.crc32(frame) & 0xFFFFFFFF)


def test_reader_prefers_read1_when_available(monkeypatch):
    """A response with read1 is read through read1 only."""
    calls = []

    class Response:
        def __init__(self):
            self.parts = iter([_frame(b'{"content": "OK"}'), b""])

        def read(self, _):
            calls.append("read")
            return next(self.parts)

        def read1(self, _):
            calls.append("read1")
            return next(self.parts)

    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_open", lambda *_, **__: Response())
    assert list(instance._events({})) == [("assistantResponseEvent", {"content": "OK"})]
    assert set(calls) == {"read1"}


def test_reader_falls_back_to_read_without_read1(monkeypatch):
    """A response lacking read1 is read through read."""
    calls = []

    class Response:
        def __init__(self):
            self.parts = iter([_frame(b'{"content": "OK"}'), b""])

        def read(self, _):
            calls.append("read")
            return next(self.parts)

    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_open", lambda *_, **__: Response())
    assert list(instance._events({})) == [("assistantResponseEvent", {"content": "OK"})]
    assert set(calls) == {"read"}
